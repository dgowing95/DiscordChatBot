"""Standalone speech service: speech-to-text and text-to-speech on the CPU.

Small FastAPI app run as its own pod/container, used by the voice sidecar
(voicesidecar/) while the bot is in a Discord voice channel:

    POST /transcribe           body: 48 kHz mono s16le PCM
                               ->  {"text", "duration", "seconds"}
    POST /speak                {"text", "voice", "speed"}
                               ->  48 kHz stereo s16le PCM (what Discord plays)
    GET  /voices               ->  {"voices": [...], "default": "..."}
    GET  /health               ->  200 once both models are loaded, 503 before

CPU only, on purpose: the GPUs are taken by llama.cpp and the image model.

  * STT is faster-whisper (CTranslate2, int8). Every utterance anyone says in
    the channel is transcribed -- that is how the wake phrase is found -- so
    the model is small and greedy (beam 1) and Silero VAD trims the silence
    the sidecar's segmenter leaves in. There is deliberately no prompt or
    hotwords hint: given the wake phrase as one, Whisper tends to leave those
    very words OUT of the transcript (measured: "Hey <name>, what
    is 12 times 12?" came back as "what is 12 times 12?"), and then the wake
    check never matches. Unhinted it spells the name well enough for the
    fuzzy match in core.
  * TTS is Kokoro-82M (ONNX). The sidecar asks for one sentence at a time and
    fetches the next while the current one plays, so the first words start
    after one short synthesis rather than the whole answer.

Configuration (all env vars optional):
  STT_MODEL          faster-whisper model (default: small.en)
  STT_COMPUTE_TYPE   CTranslate2 compute type (default: int8)
  STT_THREADS        CPU threads per transcription (default: 4)
  STT_CONCURRENCY    transcriptions run at once (default: 2)
  TTS_MODEL_URL      Kokoro ONNX model to download on first boot
  TTS_VOICES_URL     Kokoro voices file to download on first boot
  TTS_THREADS        CPU threads per synthesis (default: 4)
  TTS_CONCURRENCY    syntheses run at once (default: 2)
  VOICE_DEFAULT_VOICE  voice when a request names none (default: af_heart)
  SPEECH_QUEUE_SIZE  requests allowed to wait per kind, then 503 (default: 16)
  HF_HOME            model cache dir (/models in k8s/compose)
"""
import asyncio
import logging
import os
import time
import urllib.request
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
import soxr
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from pydantic import BaseModel

from speech_params import (
    clamp_speed,
    env_positive_int,
    env_str,
    join_segments,
    keep_segment,
    parse_voice_spec,
    voice_lang,
)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

STT_MODEL = env_str("STT_MODEL", "small.en")
STT_COMPUTE_TYPE = env_str("STT_COMPUTE_TYPE", "int8")
STT_THREADS = env_positive_int("STT_THREADS", 4)
STT_CONCURRENCY = env_positive_int("STT_CONCURRENCY", 2)
TTS_MODEL_URL = env_str(
    "TTS_MODEL_URL",
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx")
TTS_VOICES_URL = env_str(
    "TTS_VOICES_URL",
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin")
TTS_THREADS = env_positive_int("TTS_THREADS", 4)
TTS_CONCURRENCY = env_positive_int("TTS_CONCURRENCY", 2)
DEFAULT_VOICE = env_str("VOICE_DEFAULT_VOICE", "af_heart")
QUEUE_SIZE = env_positive_int("SPEECH_QUEUE_SIZE", 16)
MODEL_DIR = os.environ.get("HF_HOME") or "/models"

# The sidecar sends 48 kHz (Discord's rate, downmixed to mono); Whisper wants
# 16 kHz. Kokoro makes 24 kHz; Discord plays 48 kHz stereo.
INPUT_RATE = 48000
WHISPER_RATE = 16000
OUTPUT_RATE = 48000
# A minute of audio is far past any utterance the sidecar cuts (30 s max);
# this only stops an oversized body reaching the decoder.
MAX_INPUT_BYTES = 60 * INPUT_RATE * 2
MAX_TEXT_CHARS = 1000

whisper: WhisperModel | None = None
kokoro: Kokoro | None = None
ready = False


class Gate:
    """At most `running` jobs at once and `waiting` queued behind them;
    past that the caller gets a 503 instead of an ever-growing backlog."""

    def __init__(self, running: int, waiting: int):
        self._semaphore = asyncio.Semaphore(running)
        self._limit = running + waiting
        self._count = 0

    async def run(self, fn, *args):
        if self._count >= self._limit:
            raise HTTPException(503, "busy")
        self._count += 1
        try:
            async with self._semaphore:
                return await asyncio.to_thread(fn, *args)
        finally:
            self._count -= 1


stt_gate = Gate(STT_CONCURRENCY, QUEUE_SIZE)
tts_gate = Gate(TTS_CONCURRENCY, QUEUE_SIZE)


def _download(url: str) -> str:
    path = os.path.join(MODEL_DIR, "kokoro", os.path.basename(url))
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info(f"downloading {url}")
        partial = path + ".part"
        urllib.request.urlretrieve(url, partial)
        os.replace(partial, path)
    return path


def load_models():
    global whisper, kokoro, ready
    started = time.monotonic()
    whisper = WhisperModel(STT_MODEL, device="cpu", compute_type=STT_COMPUTE_TYPE,
                           cpu_threads=STT_THREADS, num_workers=STT_CONCURRENCY,
                           download_root=os.path.join(MODEL_DIR, "whisper"))
    options = ort.SessionOptions()
    options.intra_op_num_threads = TTS_THREADS
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(_download(TTS_MODEL_URL), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    kokoro = Kokoro.from_session(session, _download(TTS_VOICES_URL))
    if DEFAULT_VOICE not in kokoro.get_voices():
        logger.warning(f"VOICE_DEFAULT_VOICE {DEFAULT_VOICE!r} is not a Kokoro voice")
    # One throwaway synthesis and transcription so the first real request
    # does not also pay for ONNX/CTranslate2 warm-up.
    audio, _ = kokoro.create("Ready.", voice=_default_voice(), speed=1.0)
    list(whisper.transcribe(soxr.resample(audio, 24000, WHISPER_RATE), beam_size=1)[0])
    ready = True
    logger.info(f"speech models ready in {time.monotonic() - started:.1f}s "
                f"(stt={STT_MODEL}/{STT_COMPUTE_TYPE})")


def _default_voice() -> str:
    voices = kokoro.get_voices()
    return DEFAULT_VOICE if DEFAULT_VOICE in voices else voices[0]


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(asyncio.to_thread(load_models))
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    if not ready:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok", "stt_model": STT_MODEL}


@app.get("/voices")
async def voices():
    if not ready:
        raise HTTPException(503, "loading")
    return {"voices": kokoro.get_voices(), "default": _default_voice()}


def _transcribe(pcm: bytes) -> dict:
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    audio = soxr.resample(samples, INPUT_RATE, WHISPER_RATE)
    segments, info = whisper.transcribe(
        audio,
        language="en" if STT_MODEL.endswith(".en") else None,
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    texts = [s.text for s in segments if keep_segment(s.no_speech_prob, s.avg_logprob)]
    return {"text": join_segments(texts), "duration": round(info.duration, 2)}


@app.post("/transcribe")
async def transcribe(request: Request):
    if not ready:
        raise HTTPException(503, "loading")
    pcm = await request.body()
    if len(pcm) > MAX_INPUT_BYTES:
        raise HTTPException(413, "audio too long")
    if len(pcm) < 2:
        return {"text": "", "duration": 0.0, "seconds": 0.0}
    started = time.monotonic()
    result = await stt_gate.run(_transcribe, pcm[: len(pcm) // 2 * 2])
    result["seconds"] = round(time.monotonic() - started, 3)
    return result


class SpeakRequest(BaseModel):
    text: str
    voice: str | None = None
    speed: float | None = None


def _style(spec: str):
    """(voice for Kokoro, language) for a voice spec; a blend becomes the
    weighted sum of its style vectors."""
    try:
        blend = parse_voice_spec(spec, kokoro.get_voices())
    except ValueError as e:
        raise HTTPException(400, str(e)) from None
    if len(blend) == 1:
        return blend[0][0], voice_lang(blend)
    return sum(kokoro.get_voice_style(name) * weight for name, weight in blend), voice_lang(blend)


def _speak(text: str, voice, lang: str, speed: float) -> bytes:
    audio, rate = kokoro.create(text, voice=voice, speed=speed, lang=lang)
    audio = soxr.resample(audio, rate, OUTPUT_RATE)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    return np.repeat(pcm, 2).tobytes()  # interleaved stereo


@app.post("/speak")
async def speak(request: SpeakRequest):
    if not ready:
        raise HTTPException(503, "loading")
    text = request.text.strip()[:MAX_TEXT_CHARS]
    if not text:
        raise HTTPException(400, "no text")
    voice, lang = _style(request.voice or _default_voice())
    started = time.monotonic()
    pcm = await tts_gate.run(_speak, text, voice, lang, clamp_speed(request.speed or 1.0))
    return Response(pcm, media_type="application/octet-stream",
                    headers={"X-Synthesis-Seconds": f"{time.monotonic() - started:.3f}"})
