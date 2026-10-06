"""Standalone image-generation (diffusion) service.

Small FastAPI app run as its own pod/container (separate from the bot core):

    POST /generate   {"prompt": "...", "negative_prompt": "...",
                      "reference_image": "<base64>"}  ->  image/png
    GET  /health     ->  200 once the model is loaded, 503 while loading;
                         also says which prompt style the model wants and
                         whether it can edit (take a reference_image)

Two model families (generation_params.pipeline_family):
  * SD 1.5 / SDXL (e.g. RunDiffusion/Juggernaut-XL-v9): text-to-image only,
    CLIP-conditioned, so core rewrites requests into tag-list prompts plus a
    negative prompt first.
  * FLUX.2 [klein] (black-forest-labs/FLUX.2-klein-4B): a Qwen3 text encoder
    that reads plain sentences, and reference-image editing in the same model.
    It binds attributes ("obese body, stick-thin arms") and scale ("towering
    over the city") far better than CLIP can, and renders legible text. On an
    8 GB card it needs IMAGE_QUANTIZE=nf4: measured on prod's RTX 2070 that
    is 12-14 s per 1024px image (edits 22-25 s), 3.9-5.1 GB peak VRAM and
    ~9 GB of RAM.

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
                   for SD1.5, 5.0 for SDXL; pinned for distilled models: 0.0
                   for sd-turbo and friends, 1.0 for FLUX.2 [klein])
  IMAGE_NEGATIVE_PROMPT
                   baseline negative prompt merged behind the per-request one
                   (SD/SDXL only; FLUX.2 has no negative prompt)
  IMAGE_LONG_PROMPT
                   1/0: encode prompts longer than 77 CLIP tokens instead of
                   truncating them (default: 1)
  IMAGE_OFFLOAD    model | sequential | none (default: model)
  IMAGE_QUANTIZE   none | nf4 (default: none). nf4 loads FLUX.2 [klein]'s
                   transformer and text encoder in 4-bit (bitsandbytes);
                   ignored for SD/SDXL
  IMAGE_QUEUE_SIZE max queued requests, then 503 (default: 16)
  IMAGE_SEED       fixed seed for reproducible images (default: random)
  HF_HOME          model cache dir (set to /models in k8s/compose; the model
                   is downloaded once and survives redeploys on the volume)
"""
import logging
import asyncio
import base64
import binascii
import inspect
import io
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageOps
from pydantic import BaseModel

from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)

from generation_params import (
    component_dtypes,
    env_choice,
    env_flag,
    env_optional_float,
    env_optional_int,
    env_positive_int,
    env_str,
    is_distilled,
    merge_negative_prompt,
    pipeline_family,
    prompt_style,
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
    # Base64 PNG/JPEG to edit, for pipelines that take one (/health "edits").
    reference_image: str | None = None
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
QUANTIZE = env_choice("IMAGE_QUANTIZE", ("none", "nf4"), "none")
# A reference image is a Discord attachment the bot made itself (a few MB at
# most); this only stops an oversized upload from reaching the decoder.
MAX_REFERENCE_BYTES = 10 * 1024 * 1024

if os.environ.get("HF_HOME"):
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)

pipe = None
compel = None
ready = False
# Settled by load_pipeline from the repo's model_index.json.
FAMILY = "sd"
DISTILLED = is_distilled(MODEL)
DTYPE = torch.float16
EDITS = False


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


def _build_compel(p):
    """A compel encoder for an SD/SDXL pipeline, or None to keep the plain-string path.

    CLIP's context is 77 tokens (~55 words). Passing a longer prompt as a
    string means diffusers tokenizes it, truncates, and logs a warning -- so
    everything after ~55 words was thrown away, which in practice is the style,
    lighting and quality half of the prompt, because that is what gets written
    last. compel instead encodes the prompt in as many 77-token chunks as it
    needs and concatenates the embeddings (both wrappers set
    truncate_long_prompts=False, which plain Compel does not default to).

    device= is the fix for a failure on every request in prod: the
    multi-encoder Compel([...]) this replaced never passed a device down to its
    per-encoder providers, so under model CPU offload each built its token ids
    on the CPU while the encoder had been moved to cuda:0 ("index is on cpu,
    different from other tensors on cuda:0"). _encode caught that and fell
    back to truncating, so the long-prompt path had silently never worked --
    12 of 12 logged generations, and Big Sally's scale and style terms were
    exactly the part past token 77. CPU-only dev runs never showed it.
    """
    from compel import CompelForSD, CompelForSDXL

    wrapper = CompelForSDXL if isinstance(p, StableDiffusionXLPipeline) else CompelForSD
    return wrapper(p, device=p._execution_device)


def _model_index(model_id: str) -> dict:
    """The repo's model_index.json (a few KB), or {} when there is none or it
    cannot be read -- which loads the model exactly as before. Read ahead of
    the weights so the family-specific choices can be made before loading."""
    try:
        return DiffusionPipeline.load_config(model_id)
    except Exception as e:
        logger.warning(f"could not read model_index.json for {model_id!r} ({e!r}); "
                       f"assuming an SD/SDXL pipeline")
        return {}


def _nf4_config(dtypes: dict):
    """4-bit (bitsandbytes NF4) for FLUX.2 [klein]'s transformer and text
    encoder, each computing in its own dtype (component_dtypes).

    Needed to fit the 8 GB card: unquantized, the text encoder alone is 8 GB in
    16-bit. Model CPU offload works with it (4-bit modules move between devices;
    it is 8-bit ones that diffusers leaves pinned to the GPU).
    """
    from diffusers import BitsAndBytesConfig as DiffusersBnb
    from diffusers import PipelineQuantizationConfig
    from transformers import BitsAndBytesConfig as TransformersBnb

    def nf4(config_class, dtype):
        return config_class(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=dtype)

    return PipelineQuantizationConfig(quant_mapping={
        "transformer": nf4(DiffusersBnb, dtypes["default"]),
        "text_encoder": nf4(TransformersBnb, dtypes.get("text_encoder", dtypes["default"])),
    })


def _optional(label, enable):
    """Run an optional memory saver; not every pipeline class has each one."""
    try:
        enable()
    except Exception as e:
        logger.info(f"{label} not available for this pipeline ({e!r})")


def load_pipeline():
    """Load the diffusion pipeline once, at boot (blocking)."""
    global pipe, compel, ready, FAMILY, DISTILLED, DTYPE, EDITS
    has_cuda = torch.cuda.is_available()
    if not has_cuda:
        logger.info("No CUDA device found; running on CPU (slow, dev only)")
    index = _model_index(MODEL)
    FAMILY = pipeline_family(index.get("_class_name"))
    DISTILLED = is_distilled(MODEL, bool(index.get("is_distilled")))
    if has_cuda:
        bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        dtypes = {name: getattr(torch, value)
                  for name, value in component_dtypes(FAMILY, bf16).items()}
    else:
        dtypes = {"default": torch.float32}
    dtype = DTYPE = dtypes["default"]
    dtype_names = {name: str(value).removeprefix("torch.") for name, value in dtypes.items()}
    logger.info(f"Loading diffusion pipeline {MODEL!r} (family={FAMILY}, "
          f"steps={STEPS}, {WIDTH}x{HEIGHT}, offload={OFFLOAD}, quantize={QUANTIZE}, "
          f"dtypes={dtype_names}, seed={SEED})")
    # DiffusionPipeline picks the right class (SD1.5/SDXL/FLUX.2...) from the
    # repo's model_index.json, so IMAGE_MODEL can be swapped without code
    # changes. use_safetensors=None: auto-detect per component — prefer
    # safetensors when the repo ships them, fall back to .bin (e.g.
    # Lykon/DreamShaper's unet/vae/text_encoder are .bin-only).
    # safety_checker=None (SD family only; other classes would just warn that
    # they ignore it): the legacy SD1.5 NSFW checker that some repos ship (e.g.
    # DreamShaper) false-positives and blanks images to pure black; this is a
    # local/homelab deployment and the bot's own content guard covers the LLM
    # tool path.
    kwargs = {"dtype": dtypes if len(dtypes) > 1 else dtype}
    if FAMILY == "sd":
        kwargs["safety_checker"] = None
    if QUANTIZE == "nf4":
        if FAMILY == "flux2" and has_cuda:
            kwargs["quantization_config"] = _nf4_config(dtypes)
        else:
            logger.warning("IMAGE_QUANTIZE=nf4 only applies to FLUX.2 [klein] on a CUDA GPU; "
                           "loading unquantized")
    try:
        p = DiffusionPipeline.from_pretrained(MODEL, **kwargs)
    except Exception as e:
        # Fallback for hybrid repos (see _hf_single_file_path): load the
        # root-level single-file checkpoint instead. from_single_file
        # extracts unet/vae/text_encoder(s) from it and auto-detects the
        # model family; it needs no safety_checker (there is none to load).
        # SD/SDXL only: klein's repo also ships a root-level checkpoint, which
        # this path would try to load as an SD pipeline.
        single = _hf_single_file_path(MODEL) if FAMILY == "sd" else None
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
    _optional("attention slicing", lambda: p.enable_attention_slicing())
    _optional("VAE slicing", lambda: p.vae.enable_slicing())
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
    # Only the FLUX.2 path (_generate_flux2) passes a reference image on, so
    # only it may advertise editing; the signature check guards a klein
    # variant whose pipeline does not take one.
    EDITS = FAMILY == "flux2" and "image" in inspect.signature(p.__call__).parameters
    if LONG_PROMPT and FAMILY == "sd":
        try:
            compel = _build_compel(p)
        except Exception as e:
            # Never fatal: without compel the service still generates images,
            # it just truncates long prompts the way it always did.
            logger.warning(f"could not build the long-prompt encoder ({e!r}); "
                           f"prompts will be truncated at 77 CLIP tokens")
    ready = True
    guidance = resolve_guidance(MODEL, None, GUIDANCE, family=FAMILY, distilled=DISTILLED)
    logger.info(f"Diffusion pipeline ready ({type(p).__name__}, "
                f"guidance={guidance if guidance is not None else 'pipeline default'}, "
                f"long_prompt={compel is not None}, distilled={DISTILLED}, edits={EDITS}, "
                f"prompt_style={prompt_style(FAMILY)}, negative_prompt={NEGATIVE_PROMPT[:60]!r})")
    if DISTILLED:
        logger.info(f"{MODEL!r} is a distilled few-step model: guidance is pinned "
                    f"to {guidance} and negative prompts are dropped (they are inert "
                    f"without classifier-free guidance)")


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
        # One call encodes both, batched, so they come back the same length
        # even when one side spills into a second 77-token chunk.
        conditioning = compel(sanitize_for_compel(prompt),
                              negative_prompt=sanitize_for_compel(negative))
        # Under enable_model_cpu_offload the embeddings need moving (and
        # casting) to where the UNet will run before it sees them.
        device, dtype = pipe._execution_device, pipe.dtype
        kwargs = {
            "prompt_embeds": conditioning.embeds.to(device=device, dtype=dtype),
            "negative_prompt_embeds": conditioning.negative_embeds.to(device=device, dtype=dtype),
        }
        if conditioning.pooled_embeds is not None:  # SDXL
            kwargs["pooled_prompt_embeds"] = conditioning.pooled_embeds.to(device=device, dtype=dtype)
            kwargs["negative_pooled_prompt_embeds"] = conditioning.negative_pooled_embeds.to(
                device=device, dtype=dtype)
        return kwargs
    except Exception as e:
        logger.warning(f"long-prompt encoding failed ({e!r}); falling back to "
                       f"plain prompt strings, truncated at 77 CLIP tokens")
        return None


def decode_reference(data: str | None, width: int, height: int):
    """A request's base64 reference image as RGB at the output size, or None.

    Fitted (cropped to the output's aspect ratio, then resized) rather than
    stretched, so a reference made at another size keeps its proportions.
    Raises ValueError for anything that is not a decodable image.
    """
    if not data:
        return None
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError("reference_image is not valid base64") from e
    if len(raw) > MAX_REFERENCE_BYTES:
        raise ValueError(f"reference_image is over {MAX_REFERENCE_BYTES // (1024 * 1024)} MB")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        return ImageOps.fit(image.convert("RGB"), (width, height), Image.LANCZOS)
    except Exception as e:
        raise ValueError("reference_image is not a readable image") from e


def _png(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    if torch.cuda.is_available():
        # Release reserved-but-unused CUDA memory back to the driver.
        torch.cuda.empty_cache()
    return buf.getvalue()


def _generate_flux2(request: GenerateRequest, reference) -> bytes:
    """FLUX.2 [klein]: plain-sentence prompt, optional reference image to edit,
    no negative prompt (the family has none)."""
    guidance = resolve_guidance(MODEL, request.guidance_scale, GUIDANCE,
                                family=FAMILY, distilled=DISTILLED)
    kwargs = dict(num_inference_steps=STEPS, width=WIDTH, height=HEIGHT)
    if guidance is not None:
        kwargs["guidance_scale"] = guidance
    if SEED is not None:
        kwargs["generator"] = torch.Generator().manual_seed(SEED)
    # Encoded here rather than by passing prompt=: the pipeline hands the
    # embeddings to the transformer in the TEXT ENCODER's dtype, and on a card
    # without bf16 that is float32 against a float16 transformer
    # (component_dtypes). inference_mode because encode_prompt, unlike the
    # pipeline call, does not turn autograd off -- and with it on, every
    # dequantized 4-bit weight is kept for a backward pass that never comes
    # (measured: out of memory on the 8 GB card before the first image).
    with torch.inference_mode():
        prompt_embeds, _ = pipe.encode_prompt(prompt=request.prompt,
                                              device=pipe._execution_device)
    kwargs["prompt_embeds"] = prompt_embeds.to(DTYPE)
    if reference is not None:
        kwargs["image"] = [reference]
    logger.info(f"Generating ({'edit' if reference is not None else 'text-to-image'}): "
                f"prompt={len(request.prompt.split())} words, guidance={guidance}, "
                f"steps={STEPS}, {WIDTH}x{HEIGHT}")
    return _png(pipe(**kwargs).images[0])


def generate_bytes(request: GenerateRequest, reference=None) -> bytes:
    """Blocking single-image generation; runs in a worker thread."""
    assert pipe is not None, "pipeline not loaded yet"
    if FAMILY == "flux2":
        return _generate_flux2(request, reference)
    prompt = request.prompt
    negative = merge_negative_prompt(request.negative_prompt, NEGATIVE_PROMPT, DISTILLED)
    # DISTILLED, not the name alone: a repo whose model_index.json flags it
    # must get the guidance pin too, or its negative prompt is dropped while
    # guidance stays at the pipeline's 7.5/5.0.
    guidance = resolve_guidance(MODEL, request.guidance_scale, GUIDANCE,
                                family=FAMILY, distilled=DISTILLED)

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
    return _png(pipe(**kwargs).images[0])


async def worker(queue: asyncio.Queue):
    """The single consumer: images are generated strictly one at a time."""
    while True:
        request, reference, future = await queue.get()
        try:
            data = await asyncio.to_thread(generate_bytes, request, reference)
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
        "guidance": resolve_guidance(MODEL, None, GUIDANCE, family=FAMILY, distilled=DISTILLED),
        # What core reads to decide how to prompt this model and whether its
        # generate_image tool can edit (image_generation.service_capabilities).
        "prompt_style": prompt_style(FAMILY),
        "edits": EDITS,
    }


@app.post("/generate")
async def generate(request: GenerateRequest):
    prompt = (request.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="prompt must not be empty")
    request.prompt = prompt
    reference = None
    if request.reference_image:
        if not EDITS:
            raise HTTPException(status_code=422, detail=f"{MODEL} cannot edit images")
        try:
            # Decoded here, before queueing, so a bad upload is a 422 now
            # rather than a 500 after waiting its turn -- in a thread, so a
            # large decode does not stall /health and the other requests.
            reference = await asyncio.to_thread(decode_reference, request.reference_image,
                                                WIDTH, HEIGHT)
        except ValueError as e:
            logger.info(f"Rejected a reference image: {e!r} (cause: {e.__cause__!r})")
            raise HTTPException(status_code=422, detail=str(e))
        request.reference_image = None  # the decoded copy is all the worker needs

    future = asyncio.get_running_loop().create_future()
    try:
        queue.put_nowait((request, reference, future))
    except asyncio.QueueFull:
        raise HTTPException(status_code=503, detail="image queue is full, try again later")
    logger.info(f"Queued image {'edit' if reference is not None else 'generation'} "
                f"(queue depth {queue.qsize()}): {prompt[:80]!r}")
    try:
        data = await future
    except Exception as e:
        logger.warning(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"image generation failed: {e}")
    logger.info(f"Generated image ({len(data)} bytes) for: {prompt[:80]!r}")
    return Response(content=data, media_type="image/png")
