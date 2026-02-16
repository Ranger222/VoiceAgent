# Voice Agent

A low-latency, real-time AI voice companion built with a cascaded streaming architecture. This system uses local speech processing (faster-whisper for STT and Piper for TTS) combined with Mistral LLM for natural, conversational voice interactions.

## Architecture

```
User Speech --> faster-whisper STT --> Mistral LLM --> Piper TTS --> Audio Output
                     |                      |               |
                Local Inference      Sentence Chunker   Local ONNX
                     |                      |               |
                300ms VAD           Cascaded Stream    22050 Hz PCM
```

### How It Works

1. The browser captures microphone audio and streams it via WebSocket to the server
2. faster-whisper (local) transcribes speech using energy-based VAD with a 300ms silence threshold
3. Mistral LLM streams a response token-by-token
4. A micro-chunker detects natural speech boundaries and sends each phrase to TTS immediately
5. Piper TTS (local) synthesizes each chunk and streams audio back to the browser
6. The user hears the first words within approximately 1 second of finishing their sentence

### Cascaded Streaming

The key design principle is that TTS does not wait for the full LLM response. Each sentence or phrase is synthesized and played as soon as it is generated. This means the user hears audio while the LLM is still producing the rest of the response.

### Turn-Taking Design

The system prompt follows a voice-native conversation pattern rather than a chatbot pattern. The AI acknowledges the user, responds briefly with the key information, and returns the conversational turn with a follow-up question. This creates a natural back-and-forth rhythm.

## Prerequisites

- Python 3.10 or later
- macOS (tested on Apple M2 Air, 16 GB RAM) or Linux
- A Mistral API key
- A modern web browser with microphone access

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Ranger222/VoiceAgent.git
cd VoiceAgent
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit the `.env` file and add your Mistral API key:

```
MISTRAL_API_KEY=your_mistral_key_here
```

The STT and TTS models are fully local and require no API keys. They will be downloaded automatically on first run.

## Usage

### Start the Server

```bash
python server.py
```

On first launch, the server will:
- Download and cache the faster-whisper `base.en` model (approximately 150 MB)
- Download and cache the Piper `en_US-amy-medium` voice model (approximately 50 MB)
- Start the FastAPI server on `http://localhost:8000`

### Open the Web Interface

Navigate to `http://localhost:8000` in your browser. Click the orb to start speaking.

### Test the LLM Integration

```bash
python test_mistral.py
```

This sends a test message to Mistral and displays the streaming response with sentence chunking.

## Configuration

All settings are configured through the `.env` file:

| Variable | Default | Description |
|---|---|---|
| `MISTRAL_API_KEY` | (required) | Your Mistral API key |
| `MISTRAL_MODEL` | `mistral-small-latest` | Mistral model to use |
| `WHISPER_MODEL` | `base.en` | faster-whisper model size (`tiny.en`, `base.en`, `small.en`) |
| `PIPER_VOICE` | `en_US-amy-medium` | Piper voice model name |
| `VAD_THRESHOLD_MS` | `300` | Silence duration (ms) before triggering transcription |
| `MIN_CHUNK_LENGTH` | `10` | Minimum text chunk size for TTS |
| `MAX_CHUNK_LENGTH` | `200` | Maximum chunk size before force-splitting |

## Project Structure

```
VoiceAgent/
  orchestrator/
    __init__.py
    main.py                            - Main orchestration loop
    config.py                          - Configuration management
    components/
      __init__.py
      faster_whisper_stt.py            - Local STT using faster-whisper
      piper_tts.py                     - Local TTS using Piper
      mistral_llm.py                   - Mistral streaming client
      chunker.py                       - Sentence chunking logic
      deepgram_stt.py                  - Legacy cloud STT (deprecated)
      cartesia_tts.py                  - Legacy cloud TTS (deprecated)
  static/
    index.html                         - Web frontend
  docs/
    research_analysis.md               - Latency benchmarks and architecture comparison
  tests/
    test_chunker.py                    - Unit tests for sentence chunker
  server.py                            - FastAPI WebSocket server
  requirements.txt                     - Python dependencies
  .env.example                         - Environment variable template
```

## Performance

Tested on Apple M2 Air, 16 GB RAM:

| Metric | Value |
|---|---|
| STT latency | 595 to 1,084 ms |
| LLM first token | 538 to 593 ms |
| TTS first chunk | 60 to 195 ms |
| End-to-end perceived latency | 1,094 to 1,385 ms |
| Model load time (startup) | Approximately 2.5 seconds |
| RAM usage (models loaded) | Approximately 850 MB |

For detailed benchmarks and the full cloud-vs-local comparison, see [docs/research_analysis.md](docs/research_analysis.md).

## Troubleshooting

### No audio output

Ensure your browser has microphone permissions enabled. Check the browser console for WebSocket connection errors. Verify the server is running on port 8000.

### High STT latency

Lower the `VAD_THRESHOLD_MS` value in `.env`. Values below 200ms may cause false triggers on background noise.

### Mistral rate limit errors

The free tier has strict limits (approximately 5 requests per minute). The server will log rate limit errors. Consider upgrading to a paid tier for production use.

### Model download failures

On first run, the system downloads models from HuggingFace. Ensure you have internet access. Models are cached locally after the first download.

## Customization

### Change the AI Persona

Edit the `SYSTEM` prompt in `server.py`. The prompt follows a voice-native turn-taking pattern. Keep responses conversational and avoid instructing the model to use markdown or lists.

### Use a Different Voice

Change `PIPER_VOICE` in `.env`. Browse available Piper voices at: https://rhasspy.github.io/piper-samples/

### Use a Smaller or Larger Whisper Model

Change `WHISPER_MODEL` in `.env`:
- `tiny.en` - Fastest, lower accuracy
- `base.en` - Balanced (default)
- `small.en` - Best accuracy, higher resource usage

## License

MIT License

## Contributing

Contributions are welcome. Please open an issue or pull request.
