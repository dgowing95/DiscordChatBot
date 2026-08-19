"""Standalone image-generation (diffusion) service.

Small FastAPI app run as its own pod/container (separate from the bot core):

    POST /generate   {"prompt": "..."}  ->  image/png          (text-to-image)
    POST /generate   {"prompt": "...", "image": "<base64>", "strength": 0.6}
                    ->  image/png          (image-to-image / editing)
    GET  /health     ->  200 once the model is loaded, 503 while loading

The img2img pipeline is derived from the SAME loaded components (shared
unet/vae/text_encoder; only the scheduler is cloned), so it costs no extra
model download, CPU RAM or VRAM — and it runs fewer steps than txt2img
(int(steps * (1 - strength))).

Design goals:
  * As little VRAM as possible: fp16 weights and model CPU offload so only
    ONE pipeline component sits on the GPU at a time. The text encoder
    lives in CPU RAM and is moved to the GPU only while it encodes the
    prompt (sequential offload goes further: every module is offloaded, at
    the cost of speed). Attention + VAE slicing cut activation VRAM so
    SDXL at 1024x1024 is viable on an 8GB card.
  * Any diffusers-format SD1.5/SDXL repo works: the pipeline class is
    auto-detected from the repo's model_index.json (e.g.
    RunDiffusion/Juggernaut-XL-v9 for SDXL). For SDXL the legacy DDPM
    scheduler that many repos ship is upgraded to DPM++ 2M Karras (SDXL
    fine-tunes are tuned for it); distilled models (sd-turbo) keep their
    own scheduler.
  * Queued: every request is put on an asyncio queue and consumed by a
    single worker, so images are generated strictly one at a time even when
    several Discord messages ask for images at once.

Configuration (all env vars optional):
  IMAGE_MODEL      HF repo id of the pipeline (default: stabilityai/sd-turbo)
  IMAGE_STEPS      sampler steps (default: 4; sd-turbo supports 1-4)
  IMAGE_WIDTH      output width in px (default: 512)
  IMAGE_HEIGHT     output height in px (default: 512)
  IMAGE_OFFLOAD    model | sequential | none (default: model)
  IMAGE_QUEUE_SIZE max queued requests, then 503 (default: 16)
  IMAGE_SEED       fixed seed for reproducible images (default: random)
  IMAGE_EDIT_STRENGTH  default img2img strength 0<s<1 (default: 0.5)
  HF_HOME          model cache dir (set to /models in k8s/compose; the model
                   is downloaded once and survives redeploys on the volume)
"""
import asyncio
import base64
import copy
import io
import os
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from PIL import Image

from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
)


class GenerateRequest(BaseModel):
    prompt: str
    # base64-encoded source image; when set, the request becomes img2img
    image: Optional[str] = None
    # 0 < strength < 1: how far the result moves from the source image
    strength: Optional[float] = None


def _env_positive_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except ValueError:
        return default
    return value if value > 0 else default


def _env_optional_int(name: str):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


MODEL = os.environ.get("IMAGE_MODEL", "stabilityai/sd-turbo")
STEPS = _env_positive_int("IMAGE_STEPS", 4)
WIDTH = _env_positive_int("IMAGE_WIDTH", 512)
HEIGHT = _env_positive_int("IMAGE_HEIGHT", 512)
OFFLOAD = os.environ.get("IMAGE_OFFLOAD", "model").strip().lower()
QUEUE_SIZE = _env_positive_int("IMAGE_QUEUE_SIZE", 16)
SEED = _env_optional_int("IMAGE_SEED")
try:
    EDIT_STRENGTH = float(os.environ.get("IMAGE_EDIT_STRENGTH", "0.5"))
    if not 0.0 < EDIT_STRENGTH < 1.0:
        EDIT_STRENGTH = 0.5
except ValueError:
    EDIT_STRENGTH = 0.5

if os.environ.get("HF_HOME"):
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

pipe = None
img2img_pipe = None
ready = False


def _is_sdxl_checkpoint(path: str) -> bool:
    """Detect the SDXL family in a single-file checkpoint from the
    safetensors header alone (no tensor data is read).

    SDXL bundles a second (larger) text encoder; the marker key depends on
    the exporter's naming convention:
      diffusers -> text_encoder_2.*
      A1111     -> cond_stage_model_2.*
      ComfyUI   -> conditioner.embedders.1.*  (clip_g)
    """
    import json
    import struct
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    keys = [k for k in header if k != "__metadata__"]
    return (
        any(k.split(".")[0] in ("text_encoder_2", "cond_stage_model_2") for k in keys)
        or any(k.startswith("conditioner.embedders.1.") for k in keys)
    )


def _hf_single_file_path(repo_id: str):
    """Local (HF-cached) path to the repo's root-level checkpoint file.

    Some repos (e.g. RunDiffusion/Juggernaut-XL-v9) are "hybrid": a
    diffusers layout with NON-standard weight filenames
    (text_encoder/model.fp16.safetensors), which from_pretrained cannot
    resolve. Their primary artifact is a single-file checkpoint
    (juggernaut_XL_v9....safetensors); from_single_file parses it and
    auto-detects SD1.5 vs SDXL. Returns None when the repo has no
    root-level checkpoint file (or cannot be listed)."""
    from huggingface_hub import HfApi, hf_hub_download
    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
    except Exception as e:
        print(f"could not list files for {repo_id!r}: {e}")
        return None
    candidates = [
        (s.size or 0, s.rfilename)
        for s in info.siblings
        if "/" not in s.rfilename
        and s.rfilename.lower().endswith((".safetensors", ".ckpt", ".bin"))
    ]
    if not candidates:
        return None
    candidates.sort(reverse=True)  # largest first = the real checkpoint
    size, name = candidates[0]
    print(f"downloading single-file checkpoint {name} "
          f"({size / 1073741824:.1f} GB) from {repo_id!r}")
    return hf_hub_download(repo_id=repo_id, filename=name)


def build_img2img_pipeline(base):
    """Derive the img2img pipeline from the already-loaded txt2img components.

    Shares the exact same unet/vae/text_encoder modules (no extra weights,
    CPU RAM or VRAM); only the scheduler is copied (deepcopy — the diffusers
    schedulers have no clone() method, and deepcopy of a scheduler config
    object is cheap and side-effect free). Safe because the single-worker
    queue guarantees only one pipeline is ever in flight.
    Returns None for model families without a matching img2img class.
    """
    try:
        components = dict(base.components)
        components["scheduler"] = copy.deepcopy(base.scheduler)
        if isinstance(base, StableDiffusionXLPipeline):
            cls = StableDiffusionXLImg2ImgPipeline
        else:
            cls = StableDiffusionImg2ImgPipeline
        built = cls(**components)
        print("img2img pipeline ready (shares components with the txt2img pipeline)")
        return built
    except Exception as e:
        print(f"img2img not available for model {MODEL!r}: {e}")
        return None


def load_pipeline():
    """Load the diffusion pipeline once, at boot (blocking)."""
    global pipe, img2img_pipe, ready
    has_cuda = torch.cuda.is_available()
    if not has_cuda:
        print("No CUDA device found; running on CPU (slow, dev only)")
    dtype = torch.float16 if has_cuda else torch.float32
    print(f"Loading diffusion pipeline {MODEL!r} "
          f"(steps={STEPS}, {WIDTH}x{HEIGHT}, offload={OFFLOAD}, seed={SEED})")
    # DiffusionPipeline picks the right class (SD1.5/SDXL/...) from the repo's
    # model_index.json, so IMAGE_MODEL can be swapped without code changes.
    # use_safetensors=None: auto-detect per component — prefer safetensors
    # when the repo ships them, fall back to .bin (e.g. Lykon/DreamShaper's
    # unet/vae/text_encoder are .bin-only). safety_checker=None: the legacy
    # SD1.5 NSFW checker that some repos ship (e.g. DreamShaper) false-positives
    # and blanks images to pure black; this is a local/homelab deployment and
    # the bot's own content guard covers the LLM tool path.
    try:
        p = DiffusionPipeline.from_pretrained(
            MODEL, torch_dtype=dtype, safety_checker=None
        )
    except Exception as e:
        # Fallback for hybrid repos (see _hf_single_file_path): load the
        # root-level single-file checkpoint instead. from_single_file
        # extracts unet/vae/text_encoder(s) from it and auto-detects the
        # model family; it needs no safety_checker (there is none to load).
        single = _hf_single_file_path(MODEL)
        if single is None:
            raise
        print(f"from_pretrained failed ({e!r}); "
              f"using single-file checkpoint {single!r}")
        # from_single_file lives on the concrete pipeline classes and each
        # one validates that the checkpoint matches its family, so detect
        # the family up front.
        cls = StableDiffusionXLPipeline if _is_sdxl_checkpoint(single) \
            else StableDiffusionPipeline
        p = cls.from_single_file(single, torch_dtype=dtype)
    if isinstance(p, StableDiffusionXLPipeline):
        # SDXL checkpoint repos commonly ship the legacy DDPM scheduler;
        # the fine-tunes (e.g. Juggernaut XL) are tuned for a 2nd-order
        # solver with Karras sigmas, so upgrade it. from_config drops the
        # DDPM-only keys (beta schedule etc.) automatically. Deliberately
        # NOT done for other families — distilled models like sd-turbo are
        # trained with their shipped scheduler.
        p.scheduler = DPMSolverMultistepScheduler.from_config(
            p.scheduler.config, use_karras_scheduling=True
        )
        print("scheduler upgraded to DPM++ 2M Karras (SDXL)")
    # Slice attention and VAE activations: saves a few hundred MB of VRAM
    # at SDXL resolutions for a small speed cost, keeping 1024x1024 viable
    # on 8GB cards under CPU offload.
    p.enable_attention_slicing()
    p.enable_vae_slicing()
    if has_cuda and OFFLOAD != "none":
        if OFFLOAD == "sequential":
            p.enable_sequential_cpu_offload()
        else:
            # One pipeline component on the GPU at a time; the text encoder
            # (and everything else) sits in CPU RAM between phases.
            p.enable_model_cpu_offload()
    else:
        p = p.to("cuda" if has_cuda else "cpu")
    pipe = p
    img2img_pipe = build_img2img_pipeline(p)
    ready = True
    print("Diffusion pipeline ready")


def generate_bytes(prompt: str, source: bytes | None, strength: float) -> bytes:
    """Blocking single-image generation; runs in a worker thread.

    source=None -> text-to-image; otherwise image-to-image (source is
    re-encoded to the output size, so input dimensions are irrelevant)."""
    assert pipe is not None, "pipeline not loaded yet"
    kwargs = dict(prompt=prompt, num_inference_steps=STEPS, width=WIDTH, height=HEIGHT)
    if SEED is not None:
        kwargs["generator"] = torch.Generator().manual_seed(SEED)
    if source is None:
        result = pipe(**kwargs)
    else:
        assert img2img_pipe is not None, "img2img pipeline not available"
        img = Image.open(io.BytesIO(source)).convert("RGB")
        if (img.width, img.height) != (WIDTH, HEIGHT):
            img = img.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
        result = img2img_pipe(image=img, strength=strength, **kwargs)
    image = result.images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    if torch.cuda.is_available():
        # Release reserved-but-unused CUDA memory back to the driver.
        torch.cuda.empty_cache()
    return buf.getvalue()


async def worker(queue: asyncio.Queue):
    """The single consumer: images are generated strictly one at a time."""
    while True:
        prompt, source, strength, future = await queue.get()
        try:
            data = await asyncio.to_thread(generate_bytes, prompt, source, strength)
            future.set_result(data)
        except Exception as e:
            future.set_exception(e)
        finally:
            queue.task_done()


queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_SIZE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model before serving; the k8s readiness probe (/health)
    # keeps the pod out of service until it is ready.
    await asyncio.to_thread(load_pipeline)
    task = asyncio.get_running_loop().create_task(worker(queue))
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    if not ready:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ready", "model": MODEL, "queue_depth": queue.qsize()}


@app.post("/generate")
async def generate(request: GenerateRequest):
    prompt = (request.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="prompt must not be empty")

    source = None
    if request.image:
        try:
            source = base64.b64decode(request.image, validate=True)
        except Exception:
            raise HTTPException(status_code=422, detail="image must be valid base64")
        if not source:
            raise HTTPException(status_code=422, detail="image is empty")
        try:
            Image.open(io.BytesIO(source)).verify()
        except Exception:
            raise HTTPException(status_code=422, detail="image is not a decodable image")
        if img2img_pipe is None:
            raise HTTPException(
                status_code=503,
                detail="image-to-image is not supported for the configured model",
            )

    strength = EDIT_STRENGTH if request.strength is None else float(request.strength)
    if not 0.0 < strength < 1.0:
        raise HTTPException(status_code=422, detail="strength must be between 0 and 1 (exclusive)")

    future = asyncio.get_running_loop().create_future()
    try:
        queue.put_nowait((prompt, source, strength, future))
    except asyncio.QueueFull:
        raise HTTPException(status_code=503, detail="image queue is full, try again later")
    kind = "img2img " if source is not None else ""
    print(f"Queued {kind}image generation (queue depth {queue.qsize()}): {prompt[:80]!r}")
    try:
        data = await future
    except Exception as e:
        print(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"image generation failed: {e}")
    kind = "Edited " if source is not None else "Generated "
    print(f"{kind}image ({len(data)} bytes) for: {prompt[:80]!r}")
    return Response(content=data, media_type="image/png")
