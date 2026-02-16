"""
AI Voice Companion Server — Local STT/TTS, no cloud fallbacks.
faster-whisper (local STT) → Mistral LLM (streaming) → Piper TTS (local) → Browser audio
"""

import asyncio
import json
import time
import logging
import base64
import os
import re
import traceback
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("voice")

# ── Config ────────────────────────────────────────────────────────────────────
MI_KEY   = os.getenv("MISTRAL_API_KEY", "")
MI_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
PIPER_VOICE   = os.getenv("PIPER_VOICE", "en_US-amy-medium")
VAD_MS   = int(os.getenv("VAD_THRESHOLD_MS", "300"))

SYSTEM = (
    "You are a real-time voice companion speaking with a human on a phone call. "
    "Your goal is to feel natural, responsive, and conversational — not verbose. "
    "\n\nGuidelines: "
    "- Speak in short conversational sentences. "
    "- Prefer 1-2 sentences at a time. If your response exceeds 2 sentences, shorten it. "
    "- Do NOT give long explanations unless the user explicitly asks for detail. "
    "- After responding, gently invite the user to continue or ask a micro-follow-up. "
    "- Sound warm, curious, and attentive. "
    "- NEVER use markdown, asterisks, bold, italic, bullets, numbered lists, or any formatting. "
    "- Write only plain spoken text — no symbols, no special characters. "
    "- Avoid long monologues. It is better to respond quickly and briefly than perfectly and completely. "
    "- Pause conversationally instead of finishing every possible thought. "
    "\n\nConversation rhythm: "
    "1. Acknowledge what the user said first. "
    "2. Respond briefly with the key point. "
    "3. Return the turn to the user with a question or prompt. "
    "\n\nExample: "
    "User: 'Tell me about PM-Kisan.' "
    "Good: 'Oh yeah, PM-Kisan gives farmers about six thousand rupees a year in three installments. Want to know how to apply?' "
    "Bad: 'PM-Kisan is a government scheme launched in 2019 that provides...' (long paragraph)"
)

app = FastAPI()

# ── Load models at startup ────────────────────────────────────────────────────
from orchestrator.components.faster_whisper_stt import FasterWhisperSTT
from orchestrator.components.piper_tts import PiperTTS

stt_engine = FasterWhisperSTT(
    model_size=WHISPER_MODEL,
    device="auto",
    compute_type="int8",
    silence_duration_ms=VAD_MS,
)
tts_engine = PiperTTS(voice=PIPER_VOICE)


@app.on_event("startup")
async def startup():
    """Load ML models on server startup."""
    loop = asyncio.get_event_loop()
    log.info("Loading STT model (faster-whisper)...")
    await loop.run_in_executor(None, stt_engine.load_model)
    log.info("Loading TTS model (Piper)...")
    await loop.run_in_executor(None, tts_engine.load_model)
    log.info(f"✅ All models loaded. TTS sample_rate={tts_engine.sample_rate}")


# ── Serve frontend ────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return HTMLResponse((Path(__file__).parent / "static" / "index.html").read_text())


@app.get("/tts_sample_rate")
async def get_tts_sample_rate():
    """Return the TTS sample rate so the browser can adapt playback."""
    return {"sample_rate": tts_engine.sample_rate}


# ── Micro-Chunker (prosody-based, not sentence-based) ────────────────────────
class Chunker:
    """
    Aggressive micro-chunker for minimal TTS latency.
    
    Emits chunks as EARLY as possible:
      - After just 4 chars at any natural break (comma, period, !, ?)
      - After conjunctions/fillers for prosody breaks
      - Force-splits at 60 chars (never wait for long sentences)
    """
    BREAK_CHARS = set('.!?,;:—–')
    
    def __init__(self):
        self.buf = ""
    
    def feed(self, tok):
        self.buf += tok
        
        # Check for natural break points (micro-chunking)
        if len(self.buf) >= 4:
            for i in range(len(self.buf) - 1, max(2, len(self.buf) - len(tok) - 1), -1):
                if self.buf[i] in self.BREAK_CHARS:
                    s = self.buf[:i + 1].strip()
                    self.buf = self.buf[i + 1:].lstrip()
                    return s if s else None
        
        # Force-split at 60 chars for voice (don't wait for punctuation)
        if len(self.buf) > 60:
            idx = self.buf.rfind(" ", 0, 60)
            if idx < 1:
                idx = 60
            s = self.buf[:idx].strip()
            self.buf = self.buf[idx:].lstrip()
            return s if s else None
        
        return None
    
    def flush(self):
        s = self.buf.strip()
        self.buf = ""
        return s or None


# ── TTS Text Normalizer ──────────────────────────────────────────────────────
def clean_for_tts(text: str) -> str:
    """
    Strip markdown and normalize text for spoken TTS output.
    Converts LLM 'written' language → 'spoken' language.
    """
    # Remove markdown bold/italic
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    # Remove inline code
    text = re.sub(r"`(.*?)`", r"\1", text)
    # Remove markdown links [text](url) → text
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    # Remove any stray markdown headers
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Remove bullet markers
    text = re.sub(r"^[\-\*•]\s*", "", text, flags=re.MULTILINE)
    # Expand common symbols for speech
    text = text.replace("₹", " rupees ")
    text = text.replace("$", " dollars ")
    text = text.replace("%", " percent")
    text = text.replace("&", " and ")
    text = text.replace("/", " or ")
    # Remove parentheses but keep content
    text = re.sub(r"[\(\)]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Helpers ───────────────────────────────────────────────────────────────────
async def safe_send(ws, data):
    try:
        await ws.send_json(data)
    except Exception:
        pass


# ── TTS: Piper (local) ───────────────────────────────────────────────────────
async def speak(text, browser_ws, interrupt):
    """Synthesize speech with Piper and stream audio to browser."""
    t0 = time.time()
    try:
        # Run synthesis in a thread (Piper is synchronous)
        pcm_data = await tts_engine.synthesize_async(text)

        ms = int((time.time() - t0) * 1000)
        log.info(f"    TTS synthesized: {ms}ms, {len(pcm_data)} bytes")
        await safe_send(browser_ws, {"type": "metric", "name": "tts_latency_ms", "value": ms})

        if interrupt.is_set():
            return

        # Send audio in chunks to browser (each chunk ~4096 samples = ~8192 bytes)
        chunk_size = 8192
        for i in range(0, len(pcm_data), chunk_size):
            if interrupt.is_set():
                break
            chunk = pcm_data[i:i + chunk_size]
            b64 = base64.b64encode(chunk).decode("ascii")
            await safe_send(browser_ws, {"type": "audio", "data": b64})
            # Small yield to allow other tasks to run
            await asyncio.sleep(0.001)

    except Exception as e:
        log.error(f"TTS error: {e}")


# ── LLM: Mistral ─────────────────────────────────────────────────────────────
async def think(text, history, browser_ws, interrupt):
    """Stream Mistral → chunks → Piper TTS → browser."""
    from mistralai import Mistral

    t0 = time.time()
    mi = Mistral(api_key=MI_KEY)
    ch = Chunker()
    msgs = [{"role": "system", "content": SYSTEM}] + history + [{"role": "user", "content": text}]
    full = ""
    ci = 0
    first = True

    try:
        stream = mi.chat.stream(model=MI_MODEL, messages=msgs, max_tokens=150, temperature=0.7)

        for ev in stream:
            if interrupt.is_set():
                break
            delta = ev.data.choices[0].delta.content if ev.data.choices else None
            if not delta:
                continue
            if first:
                ms = int((time.time()-t0)*1000)
                log.info(f"  LLM first-token: {ms}ms")
                await safe_send(browser_ws, {"type": "metric", "name": "llm_latency_ms", "value": ms})
                first = False
            full += delta
            await safe_send(browser_ws, {"type": "llm_token", "token": delta})
            sent = ch.feed(delta)
            if sent and not interrupt.is_set():
                ci += 1
                clean = clean_for_tts(sent)
                if clean:
                    log.info(f"  Chunk {ci}: {clean}")
                    await safe_send(browser_ws, {"type": "sentence", "text": clean, "index": ci})
                    await speak(clean, browser_ws, interrupt)

        rem = ch.flush()
        if rem and not interrupt.is_set():
            ci += 1
            clean = clean_for_tts(rem)
            if clean:
                log.info(f"  Chunk {ci} (flush): {clean}")
                await safe_send(browser_ws, {"type": "sentence", "text": clean, "index": ci})
                await speak(clean, browser_ws, interrupt)

        ms = int((time.time()-t0)*1000)
        log.info(f"  Total: {ms}ms, {ci} chunks")
        await safe_send(browser_ws, {"type": "metric", "name": "total_response_ms", "value": ms})
        await safe_send(browser_ws, {"type": "done", "full_response": full})
        return full

    except Exception as e:
        err = str(e)
        log.error(f"LLM error: {err}")
        if "429" in err:
            await safe_send(browser_ws, {"type": "error", "message": "Rate limited — wait a moment and try again"})
        else:
            await safe_send(browser_ws, {"type": "error", "message": f"LLM error: {err}"})
        return ""


# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(browser: WebSocket):
    await browser.accept()
    log.info("Browser connected")

    history = []
    interrupt = asyncio.Event()

    # Per-connection STT state (separate audio buffer)
    stt_local = FasterWhisperSTT(
        model_size=WHISPER_MODEL,
        device="auto",
        compute_type="int8",
        silence_duration_ms=VAD_MS,
    )
    # Share the already-loaded model
    stt_local.model = stt_engine.model

    await safe_send(browser, {"type": "status", "message": "All systems ready ✓ (local STT + TTS)"})

    # Process audio and transcription in a background task
    processing_lock = asyncio.Lock()

    async def process_transcript(transcript):
        """Process a transcript through LLM and TTS."""
        nonlocal history
        async with processing_lock:
            log.info(f"User Final: \"{transcript}\"")
            interrupt.set()
            await asyncio.sleep(0.05)
            interrupt.clear()

            await safe_send(browser, {"type": "processing"})
            resp = await think(transcript, history, browser, interrupt)
            if resp:
                history.append({"role": "user", "content": transcript})
                history.append({"role": "assistant", "content": resp})
                if len(history) > 20:
                    history = history[-20:]

    # Main loop: receive audio from browser → local STT → LLM → TTS
    bytes_received = 0
    start_time = time.time()
    loop = asyncio.get_event_loop()

    try:
        while True:
            try:
                msg = await browser.receive()
            except WebSocketDisconnect:
                break

            if msg.get("type") == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"]:
                audio_data = msg["bytes"]
                bytes_received += len(audio_data)

                if time.time() - start_time > 5:
                    log.info(f"Audio flowing: received {bytes_received} bytes")
                    bytes_received = 0
                    start_time = time.time()

                # Feed audio to local STT (run in executor since it might transcribe)
                transcript = await loop.run_in_executor(
                    None, stt_local.feed_audio, audio_data
                )

                if transcript:
                    # Send transcript to browser
                    await safe_send(browser, {
                        "type": "transcript", "text": transcript,
                        "is_final": True, "speech_final": True
                    })

                    # Measure STT latency
                    stt_ms = int((time.time() - stt_local._last_speech_time) * 1000) if stt_local._last_speech_time else 0
                    await safe_send(browser, {"type": "metric", "name": "stt_latency_ms", "value": stt_ms})

                    # Process through LLM
                    asyncio.create_task(process_transcript(transcript))

            elif "text" in msg and msg["text"]:
                try:
                    j = json.loads(msg["text"])
                    if j.get("type") == "interrupt":
                        interrupt.set()
                except Exception:
                    pass

    except Exception as e:
        log.error(f"WS loop: {e}\n{traceback.format_exc()}")
    finally:
        stt_local.clear_buffer()
        log.info("Browser disconnected")


if __name__ == "__main__":
    import uvicorn
    log.info(f"Keys: MI={'✓' if MI_KEY else '✗'}")
    log.info(f"Config: whisper={WHISPER_MODEL}, piper={PIPER_VOICE}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
