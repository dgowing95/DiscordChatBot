// HTTP client for the speech service (speechservice/main.py).

const SPEECH_URL = (process.env.SPEECH_URL || "http://speech:8000").replace(/\/+$/, "");
const TIMEOUT_MS = 60_000;

async function request(path, options) {
  const response = await fetch(SPEECH_URL + path, { ...options, signal: AbortSignal.timeout(TIMEOUT_MS) });
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(`speech service ${path}: ${response.status} ${detail.slice(0, 200)}`);
  }
  return response;
}

/** {text, duration, seconds} for 48 kHz mono s16le PCM. */
export async function transcribe(mono) {
  const response = await request("/transcribe", {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: mono,
  });
  return response.json();
}

/** {pcm, seconds}: 48 kHz stereo s16le for one sentence. */
export async function synthesize({ text, voice, speed }) {
  const response = await request("/speak", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, voice: voice || null, speed: speed || null }),
  });
  const pcm = Buffer.from(await response.arrayBuffer());
  return { pcm, seconds: Number(response.headers.get("x-synthesis-seconds")) || 0 };
}
