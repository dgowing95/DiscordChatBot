"""Standalone image-generation (diffusion) service.

Small FastAPI app run as its own pod/container (separate from the bot core):

    POST /generate   {"prompt": "..."}  ->  image/png          (text-to-image)
    GET  /health     ->  200 once the model is loaded, 503 while loading

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
  * Prompt adherence: CLIP truncates at 77 tokens, which silently dropped
    the tail of anything longer than ~55 words -- the style and quality
    terms, since those are written last. compel encodes the prompt in as
    many 77-token chunks as it needs and the embeddings go to the pipeline
    directly, so nothing is cut. A negative prompt and an explicit guidance
    scale are plumbed through too; both are no-ops on distilled models,
    which generation_params.py handles.
  * Queued: every request is put on an asyncio queue and consumed by a
    single worker, so images are generated strictly one at a time even when
    several Discord messages ask for images at once.

Configuration (all env vars optional):
  IMAGE_MODEL      HF repo id of the pipeline (default: stabilityai/sd-turbo)
  IMAGE_STEPS      sampler steps (default: 4; sd-turbo supports 1-4)
  IMAGE_WIDTH      output width in px (default: 512)
  IMAGE_HEIGHT     output height in px (default: 512)
  IMAGE_GUIDANCE   CFG scale (default: unset, i.e. the pipeline's own -- 7.5
                   for SD1.5, 5.0 for SDXL; forced to 0.0 for distilled models)
  IMAGE_NEGATIVE_PROMPT
                   baseline negative prompt merged behind the per-request one
  IMAGE_LONG_PROMPT
                   1/0: encode prompts longer than 77 CLIP tokens instead of
                   truncating them (default: 1)
  IMAGE_OFFLOAD    model | sequential | none (default: model)
  IMAGE_QUEUE_SIZE max queued requests, then 503 (default: 16)
  IMAGE_SEED       fixed seed for reproducible images (default: random)
  HF_HOME          model cache dir (set to /models in k8s/compose; the model
                   is downloaded once and survives redeploys on the volume)
"""
import logging
import asyncio
import io
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)

from generation_params import (
    env_flag,
    env_optional_float,
    env_optional_int,
    env_positive_int,
    env_str,
    is_distilled,
    merge_negative_prompt,
    resolve_guidance,
    sanitize_for_compel,
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    guidance_scale: float | None = None
    # Steps and dimensions are deliberately NOT per-request: they set how long
    # the single worker is occupied, and this endpoint is reachable from any
    # Discord user via the generate_image tool.


MODEL = os.environ.get("IMAGE_MODEL", "stabilityai/sd-turbo")
STEPS = env_positive_int("IMAGE_STEPS", 4)
WIDTH = env_positive_int("IMAGE_WIDTH", 512)
HEIGHT = env_positive_int("IMAGE_HEIGHT", 512)
OFFLOAD = os.environ.get("IMAGE_OFFLOAD", "model").strip().lower()
QUEUE_SIZE = env_positive_int("IMAGE_QUEUE_SIZE", 16)
SEED = env_optional_int("IMAGE_SEED")
GUIDANCE = env_optional_float("IMAGE_GUIDANCE")
NEGATIVE_PROMPT = env_str("IMAGE_NEGATIVE_PROMPT")
LONG_PROMPT = env_flag("IMAGE_LONG_PROMPT", True)
DISTILLED = is_distilled(MODEL)

if os.environ.get("HF_HOME"):
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

pipe = None
compel = None
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
        logger.warning(f"could not list files for {repo_id!r}: {e}")
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
    logger.info(f"downloading single-file checkpoint {name} "
          f"({size / 1073741824:.1f} GB) from {repo_id!r}")
    return hf_hub_download(repo_id=repo_id, filename=name)


def _build_compel(p, dtype):
    """A compel encoder for the pipeline, or None to keep the plain-string path.

    CLIP's context is 77 tokens (~55 words). Passing a longer prompt as a
    string means diffusers tokenizes it, truncates, and logs a warning -- so
    everything after ~55 words was thrown away, which in practice is the style,
    lighting and quality half of the prompt, because that is what gets written
    last. compel instead encodes the prompt in as many 77-token chunks as it
    needs and concatenates the embeddings.

    truncate_long_prompts=False is the whole point of this function: compel
    defaults it to True, and left at the default it truncates at 77 exactly
    like the pipeline does.
    """
    from compel import Compel, ReturnedEmbeddingsType

    # dtype_for_device_getter drives the padding conditioning compel builds
    # for the shorter of the two prompts; left at its float32 default it does
    # not match the fp16 weights.
    common = dict(truncate_long_prompts=False, dtype_for_device_getter=lambda device: dtype)
    if isinstance(p, StableDiffusionXLPipeline):
        # SDXL conditions on both encoders concatenated, and additionally on
        # the SECOND encoder's pooled output (requires_pooled=[False, True]),
        # taken from the penultimate hidden layer.
        return Compel(
            tokenizer=[p.tokenizer, p.tokenizer_2],
            text_encoder=[p.text_encoder, p.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            **common,
        )
    return Compel(tokenizer=p.tokenizer, text_encoder=p.text_encoder, **common)


def load_pipeline():
    """Load the diffusion pipeline once, at boot (blocking)."""
    global pipe, compel, ready
    has_cuda = torch.cuda.is_available()
    if not has_cuda:
        logger.info("No CUDA device found; running on CPU (slow, dev only)")
    dtype = torch.float16 if has_cuda else torch.float32
    logger.info(f"Loading diffusion pipeline {MODEL!r} "
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
        logger.warning(f"from_pretrained failed ({e!r}); "
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
        #
        # The kwarg is use_karras_SIGMAS. This read use_karras_scheduling,
        # which DPMSolverMultistepScheduler does not take and which
        # ConfigMixin.from_config discards without raising — so the sigmas
        # stayed at their default for as long as the log line below claimed
        # they were Karras.
        p.scheduler = DPMSolverMultistepScheduler.from_config(
            p.scheduler.config, use_karras_sigmas=True
        )
        logger.info("scheduler upgraded to DPM++ 2M Karras (SDXL)")
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
    if LONG_PROMPT:
        try:
            compel = _build_compel(p, dtype)
        except Exception as e:
            # Never fatal: without compel the service still generates images,
            # it just truncates long prompts the way it always did.
            logger.warning(f"could not build the long-prompt encoder ({e!r}); "
                           f"prompts will be truncated at 77 CLIP tokens")
    ready = True
    logger.info(f"Diffusion pipeline ready (guidance={GUIDANCE if GUIDANCE is not None else 'pipeline default'}, "
                f"long_prompt={compel is not None}, distilled={DISTILLED}, "
                f"negative_prompt={NEGATIVE_PROMPT[:60]!r})")
    if DISTILLED:
        logger.info(f"{MODEL!r} is a distilled few-step model: guidance is forced "
                    f"to 0.0 and negative prompts are dropped (they are inert "
                    f"below CFG 1)")


def _token_count(text: str) -> int:
    """What `text` costs in CLIP tokens, marker tokens included.

    Logged per request: anything over 77 is what the plain-string path would
    have silently thrown away, so this is the one number that says whether the
    long-prompt encoder is earning its keep.
    """
    try:
        return len(pipe.tokenizer(text or "").input_ids)
    except Exception:
        return -1


def _encode(prompt: str, negative: str):
    """compel conditioning tensors as pipeline kwargs, or None to fall back.

    Never raises. A failure here -- a prompt compel's parser chokes on, a
    device or dtype mismatch under CPU offload -- costs the tokens past 77,
    not the image.
    """
    try:
        positive_cond = compel(sanitize_for_compel(prompt))
        negative_cond = compel(sanitize_for_compel(negative))
        pooled = negative_pooled = None
        if isinstance(positive_cond, tuple):  # SDXL: (conditioning, pooled)
            positive_cond, pooled = positive_cond
            negative_cond, negative_pooled = negative_cond
        # The two must agree on sequence length, and they will not as soon as
        # either side spills into a second 77-token chunk.
        positive_cond, negative_cond = compel.pad_conditioning_tensors_to_same_length(
            [positive_cond, negative_cond]
        )
        # Under enable_model_cpu_offload the text encoders live in CPU RAM and
        # accelerate's hooks hand their output back on the input device, so the
        # embeddings need moving (and casting -- compel builds its padding in
        # the dtype it was given) before the UNet sees them.
        device, dtype = pipe._execution_device, pipe.dtype
        kwargs = {
            "prompt_embeds": positive_cond.to(device=device, dtype=dtype),
            "negative_prompt_embeds": negative_cond.to(device=device, dtype=dtype),
        }
        if pooled is not None:
            kwargs["pooled_prompt_embeds"] = pooled.to(device=device, dtype=dtype)
            kwargs["negative_pooled_prompt_embeds"] = negative_pooled.to(device=device, dtype=dtype)
        return kwargs
    except Exception as e:
        logger.warning(f"long-prompt encoding failed ({e!r}); falling back to "
                       f"plain prompt strings, truncated at 77 CLIP tokens")
        return None


def generate_bytes(request: GenerateRequest) -> bytes:
    """Blocking single-image generation (text-to-image); runs in a worker thread."""
    assert pipe is not None, "pipeline not loaded yet"
    prompt = request.prompt
    negative = merge_negative_prompt(request.negative_prompt, NEGATIVE_PROMPT, DISTILLED)
    guidance = resolve_guidance(MODEL, request.guidance_scale, GUIDANCE)

    kwargs = dict(num_inference_steps=STEPS, width=WIDTH, height=HEIGHT)
    if guidance is not None:
        # Omitted entirely when unresolved, so the pipeline class keeps
        # supplying its own default (7.5 SD1.5 / 5.0 SDXL).
        kwargs["guidance_scale"] = guidance
    if SEED is not None:
        kwargs["generator"] = torch.Generator().manual_seed(SEED)

    conditioning = _encode(prompt, negative) if compel is not None else None
    if conditioning is not None:
        kwargs.update(conditioning)
        encoded = conditioning["prompt_embeds"].shape[1]
    else:
        kwargs["prompt"] = prompt
        if negative:
            kwargs["negative_prompt"] = negative
        encoded = 0
    logger.info(f"Generating: prompt={_token_count(prompt)} CLIP tokens, "
                f"negative={_token_count(negative)}, guidance={guidance}, "
                f"steps={STEPS}, {WIDTH}x{HEIGHT}, "
                f"conditioning={f'{encoded} embeddings' if encoded else 'truncated at 77'}")
    result = pipe(**kwargs)
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
        request, future = await queue.get()
        try:
            data = await asyncio.to_thread(generate_bytes, request)
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
    return {
        "status": "ready",
        "model": MODEL,
        "queue_depth": queue.qsize(),
        "long_prompt": compel is not None,
        "guidance": resolve_guidance(MODEL, None, GUIDANCE),
    }


@app.post("/generate")
async def generate(request: GenerateRequest):
    prompt = (request.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="prompt must not be empty")
    request.prompt = prompt

    future = asyncio.get_running_loop().create_future()
    try:
        queue.put_nowait((request, future))
    except asyncio.QueueFull:
        raise HTTPException(status_code=503, detail="image queue is full, try again later")
    logger.info(f"Queued image generation (queue depth {queue.qsize()}): {prompt[:80]!r}")
    try:
        data = await future
    except Exception as e:
        logger.warning(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"image generation failed: {e}")
    logger.info(f"Generated image ({len(data)} bytes) for: {prompt[:80]!r}")
    return Response(content=data, media_type="image/png")
