# Voice Agent Architecture: Cloud vs Local Pipeline -- Comparative Analysis

## 1. Research Overview

This report presents a comparative analysis of two voice agent architectures tested on an Apple M2 Air (16 GB RAM), evaluating latency, response quality, and real-time conversational performance.

| | Architecture A (Cloud) | Architecture B (Local) |
|---|---|---|
| STT | Deepgram (cloud WebSocket) | faster-whisper base.en (local, int8) |
| LLM | Mistral mistral-small-latest (cloud API) | Mistral mistral-small-latest (cloud API) |
| TTS | Cartesia Sonic (cloud WebSocket) | Piper en_US-amy-medium (local ONNX) |

The LLM component (Mistral) remained identical across both architectures to isolate the impact of local vs cloud STT/TTS.

---

## 2. Latency Benchmarks

### 2.1 Component-Level Latency (Averaged Across Test Queries)

| Pipeline Stage | Cloud (Deepgram + Cartesia) | Local (faster-whisper + Piper) | Delta |
|---|---|---|---|
| STT latency | 3,406 ms | 595-1,084 ms | -68% to -82% |
| LLM first token | 529 ms | 538-593 ms | Approximately the same |
| TTS first byte | 141 ms | 60-195 ms | Approximately the same |
| Total response | 869 ms (post-STT only) | 1,094-1,385 ms (end-to-end) | See below |

### 2.2 End-to-End Perceived Latency

| Metric | Cloud | Local (Initial) | Local (Optimized) |
|---|---|---|---|
| User stops speaking to first audio heard | Approximately 4,275 ms | Approximately 2,400 ms | Approximately 1,100 ms |
| STT silence threshold | 800 ms (Deepgram endpointing) | 800 ms (manual VAD) | 300 ms (tuned VAD) |
| Turn-taking feel | Noticeable delay | Moderate delay | Conversational |

Key Insight: The cloud architecture's 3.4s STT latency was hidden by Deepgram's endpointing behavior. The local architecture exposed this as the primary bottleneck, enabling targeted optimization.

### 2.3 Latency by Query Type (Local Optimized Architecture)

| Query Type | Example | STT (ms) | LLM First Token (ms) | TTS First Chunk (ms) | Total (ms) | Chunks |
|---|---|---|---|---|---|---|
| Short greeting | "Hello" | 595 | 538 | 60 | 1,119 | 2-3 |
| Simple question | "Yes, please" | 595 | 593 | 97 | 1,119 | 3-4 |
| Medium question | "What is eligibility criteria for this?" | 667 | 538 | 115 | 1,385 | 4 |
| Complex question | "I don't own land, I work on rented land" | 975 | 579 | 78 | 1,094 | 4 |
| Follow-up | "Who will verify if I am a farmer?" | 708 | 593 | 195 | 1,218 | 2 |
| Long utterance | "Can you list schemes the government launched for farmers?" | 903 | 643 | 154 | 1,379 | 3-5 |

---

## 3. Optimization Journey

Four progressive optimization phases were applied to the local architecture:

### Phase 1: Direct Cloud-to-Local Migration

Replaced Deepgram WebSocket with faster-whisper base.en (int8 quantization). Replaced Cartesia WebSocket with Piper ONNX inference.

Result: Functional pipeline but STT latency remained high (3.4s) due to conservative 800ms silence threshold.

### Phase 2: STT Silence Threshold Reduction (800ms to 300ms)

Implemented adaptive noise floor tracking. Added minimum speech duration filter (250ms) to prevent noise triggers. Switched RMS calculation from Python loops to numpy vectorization.

Result: STT latency dropped to approximately 650ms average. Perceived turn-taking became noticeably faster.

### Phase 3: Micro-Chunking for TTS

Chunker now emits after just 4 characters at any natural break point (period, comma, exclamation, question mark, semicolon, colon). Force-splits at 60 characters (previously 180) to prevent TTS buffer delay. First audio chunk heard within approximately 100ms of LLM generating text.

Result: Perceived response time reduced by approximately 400ms.

### Phase 4: Voice-Optimized System Prompt

Replaced rigid word and token limits with a turn-taking conversation pattern. Prompt biases toward interaction density over information density (acknowledge, respond briefly, return the turn). Added TTS text normalizer to strip markdown formatting before speech synthesis.

Result: Natural 2-4 chunk responses instead of 15-20 chunk monologues. Responses became informative yet concise.

---

## 4. Architecture Comparison

### 4.1 Advantages of Local Architecture

| Dimension | Benefit |
|---|---|
| Privacy | Audio never leaves the device (STT and TTS are fully local) |
| Cost | Zero per-request cloud API fees for STT and TTS |
| Offline capability | STT and TTS work without internet (LLM still requires API) |
| Latency control | Silence threshold, chunking, and VAD are fully tunable |
| No rate limits | No cloud throttling on STT or TTS |

### 4.2 Advantages of Cloud Architecture

| Dimension | Benefit |
|---|---|
| Voice quality | Cartesia produces more expressive, human-like speech |
| Accuracy | Deepgram handles accents and noisy environments better |
| No local compute | No GPU or CPU load from model inference |
| Streaming STT | True real-time interim transcripts (not batch-and-wait) |

### 4.3 Resource Usage (M2 Air, 16 GB RAM)

| Resource | Cloud Architecture | Local Architecture |
|---|---|---|
| RAM usage | Approximately 120 MB (API clients only) | Approximately 850 MB (Whisper + Piper models) |
| CPU during STT | Minimal | Approximately 40% single-core (during transcription) |
| Model load time | N/A | Approximately 2.5s total (Whisper 1.1s + Piper 1.5s) |
| Disk (models) | 0 MB | Approximately 200 MB (cached after first download) |

---

## 5. Key Findings

### Finding 1: Turn-Taking Detection is the Primary Bottleneck

Voice assistant perceived latency is dominated by when the system decides the user has finished speaking, not by model inference speed. Reducing the silence threshold from 800ms to 300ms cut perceived latency by over 2 seconds.

### Finding 2: Cascaded Streaming Masks Generation Time

With micro-chunking, TTS begins playing the first phrase (for example, "Oh, sure.") while the LLM is still generating subsequent sentences. Users perceive response time as time-to-first-audio, not total generation time.

### Finding 3: Prompt Design Matters More Than Token Limits

Rigid token caps (max_tokens=80) produced abrupt, unusable responses. A turn-taking prompt pattern (acknowledge, respond, return turn) naturally produced concise, conversational replies without artificial truncation.

### Finding 4: Local TTS Requires a Text Normalization Layer

Local TTS engines read text literally. Markdown formatting (bold markers, asterisks), symbols (rupee sign, percent sign), and special characters must be normalized to spoken language before synthesis.

---

## 6. Latency Breakdown Visualization

```
Cloud Architecture (Deepgram + Cartesia):

  STT (Deepgram endpointing)     ████████████████████  3,406ms
  LLM first token                ███                     529ms
  TTS first byte (Cartesia)      █                       141ms
  ─────────────────────────────
  TOTAL perceived                ████████████████████  ~4,076ms


Local Architecture (Optimized):

  STT (faster-whisper + VAD)     ████                    650ms
  LLM first token                ███                     560ms
  TTS first byte (Piper)         █                       100ms
  ─────────────────────────────
  TOTAL perceived                ███████               ~1,200ms

  Reduction: approximately 70% improvement in perceived latency
```

---

## 7. Performance Tier Comparison

| System | Typical Perceived Latency |
|---|---|
| Amazon Alexa (legacy) | 2,000-3,000 ms |
| Typical API-based voice bots | 1,500-2,000 ms |
| This system (cloud) | Approximately 4,000 ms |
| This system (local, optimized) | Approximately 1,100-1,400 ms |
| Google Gemini Live | Approximately 600 ms |
| OpenAI Realtime API | 400-700 ms |

---

## 8. Technical Configuration

### Models Used

- STT: Systran/faster-whisper-base.en (CTranslate2, int8 quantization)
- LLM: mistral-small-latest via Mistral API (streaming)
- TTS: rhasspy/piper-voices/en_US-amy-medium (ONNX runtime)

### Key Parameters

| Parameter | Value | Impact |
|---|---|---|
| VAD_THRESHOLD_MS | 300 ms | Turn-taking sensitivity |
| silence_threshold (RMS) | 0.008 | Speech vs silence detection |
| min_speech_ms | 250 ms | Noise rejection |
| beam_size | 1 (greedy) | STT speed over accuracy |
| vad_filter | enabled | Whisper internal VAD cleanup |
| max_tokens | 150 | Response length cap |
| Chunk force-split | 60 chars | TTS micro-chunking |
| TTS sample rate | 22,050 Hz | Piper native output |

### Test Environment

- Hardware: Apple M2 Air, 16 GB RAM
- OS: macOS
- Python: 3.10
- Browser: Chrome (WebSocket + Web Audio API)

---

## 9. Recommendations for Future Work

1. Streaming STT: Replace batch-transcribe-on-silence with true streaming (for example, Whisper streaming or Moonshine) to enable speculative LLM starts during speech.

2. Higher-quality TTS: Evaluate Coqui XTTS or Bark for more expressive local voices while maintaining acceptable latency.

3. Turn-taking prediction: Implement prosody-based speech endpoint detection using pitch contour analysis, reducing silence threshold further without false triggers.

4. Local LLM: Replace Mistral API with a quantized local model (for example, Mistral 7B via llama.cpp) to eliminate the approximately 550ms network round-trip for first token.

5. Interrupt handling: Enable barge-in so users can interrupt the AI mid-sentence with immediate audio cancellation.

---

## 10. Conclusion

Migrating from cloud-based STT/TTS (Deepgram + Cartesia) to local alternatives (faster-whisper + Piper) on consumer hardware (M2 Air) achieved an approximately 70% reduction in perceived latency (4,076ms to 1,200ms) through a combination of:

- Local inference eliminating network round-trips for STT and TTS
- Aggressive silence threshold tuning (800ms to 300ms)
- Micro-chunking enabling cascaded TTS streaming
- Voice-optimized prompt engineering for natural turn-taking

The local architecture trades voice expressiveness for privacy, cost savings, and full latency control. The remaining bottleneck is the cloud LLM API call (approximately 550ms first token), which could be addressed by local LLM inference in future iterations.
