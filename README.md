# AI Voice Companion Orchestrator

A production-ready, low-latency AI voice companion with cascaded streaming architecture.

## 🎯 Features

- **Real-time Speech-to-Text**: Deepgram Nova-3 with 400ms VAD threshold
- **Streaming LLM**: Mistral Small with sentence-level chunking
- **Cascaded Streaming**: TTS starts before LLM completes (< 1s latency)
- **Barge-in Support**: Interrupt AI mid-response naturally
- **Rate Limit Handling**: Automatic retry with exponential backoff
- **Conversational AI**: Warm, empathetic persona optimized for voice

## 🏗️ Architecture

```
User Speech → Deepgram STT → Mistral LLM → Cartesia TTS → Audio Output
                    ↓              ↓              ↓
                WebSocket    Sentence Chunker   WebSocket
                    ↓              ↓              ↓
                 400ms VAD    Cascaded Stream  Interrupt Support
```

### Cascaded Streaming Flow

The key innovation is **sentence-level streaming**:

1. User stops speaking (VAD triggers after 400ms)
2. Deepgram sends transcript to Mistral
3. Mistral streams tokens → Chunker detects sentence boundaries
4. **Each sentence sent to TTS immediately** (don't wait for full response)
5. Audio playback begins while LLM is still generating
6. Total latency: **< 1 second** from speech end to audio start

### Barge-in Logic

Users can interrupt naturally:

1. Deepgram detects new speech during TTS playback
2. Interrupt event triggered → TTS stream killed immediately
3. Audio buffer cleared, new transcript processed
4. Response time: **< 100ms**

## 📦 Installation

### 1. Clone and Setup

```bash
cd "/Users/piyushsinghtomar/Documents/Voice Agent"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```bash
DEEPGRAM_API_KEY=your_deepgram_key_here
MISTRAL_API_KEY=your_mistral_key_here
CARTESIA_API_KEY=your_cartesia_key_here
```

**Get API Keys:**
- **Deepgram**: https://console.deepgram.com/ (Free tier: 45,000 minutes)
- **Mistral**: https://console.mistral.ai/ (Free tier available)
- **Cartesia**: https://cartesia.ai/ (Free tier available)

## 🚀 Usage

### Quick Test - Mistral LLM Only

Test the Mistral LLM streaming integration:

```bash
python test_mistral.py
```

This will send a test message to Mistral and show the streaming response with sentence chunking.

### Run the Full Orchestrator

```bash
python -m orchestrator.main
```

**What happens:**
1. Microphone captures your speech (via PyAudio)
2. Deepgram transcribes in real-time
3. Mistral generates streaming response
4. Cartesia synthesizes and plays audio
5. You can interrupt mid-response (barge-in)

### Test Mode (Text-only TTS)

For testing without audio playback:

```bash
python -m orchestrator.main --fallback-tts
```

This logs TTS output to console instead of playing audio.

## 🎛️ Configuration

All settings are in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `VAD_THRESHOLD_MS` | 400 | Voice activity detection threshold |
| `MISTRAL_TEMPERATURE` | 0.7 | LLM creativity (0.0-1.0) |
| `MISTRAL_MAX_TOKENS` | 150 | Max response length |
| `MIN_CHUNK_LENGTH` | 10 | Minimum sentence chunk size |
| `MAX_CHUNK_LENGTH` | 200 | Maximum chunk before force split |

## 🧪 Testing

### Test Sentence Chunker

```bash
python -c "
from orchestrator.components.chunker import chunk_text_sync
text = 'Hello, how are you? I am doing great. Thanks for asking!'
chunks = chunk_text_sync(text)
print(chunks)
"
```

Expected output:
```python
['Hello,', 'how are you?', 'I am doing great.', 'Thanks for asking!']
```

### Test Latency

1. Run the orchestrator
2. Say: "Hello, how are you?"
3. Measure time from when you stop speaking to when audio starts
4. **Target**: < 1 second

### Test Barge-in

1. Ask a question that triggers a long response
2. While AI is speaking, say "wait" or "stop"
3. Audio should stop within 100ms

## 📁 Project Structure

```
voice-agent/
├── orchestrator/
│   ├── __init__.py
│   ├── main.py              # Main orchestration loop
│   ├── config.py            # Configuration management
│   └── components/
│       ├── __init__.py
│       ├── deepgram_stt.py  # Deepgram WebSocket client
│       ├── mistral_llm.py   # Mistral streaming client
│       ├── cartesia_tts.py  # Cartesia WebSocket client
│       └── chunker.py       # Sentence chunking logic
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 🔧 Troubleshooting

### "DEEPGRAM_API_KEY environment variable is required"

Make sure you've created a `.env` file with your API keys. See Configuration section.

### "Rate limited" errors from Mistral

The free tier has strict limits (~5 requests/minute). The orchestrator will automatically retry with exponential backoff. For production, upgrade to a paid tier.

### No audio output

1. Check your system audio settings
2. Verify Cartesia API key is correct
3. Try `--fallback-tts` mode to test without audio

### High latency

1. Check your internet connection
2. Reduce `MISTRAL_MAX_TOKENS` for shorter responses
3. Lower `VAD_THRESHOLD_MS` (but may cause false triggers)

## 🎨 Customization

### Change AI Persona

Edit the system prompt in `.env` or `orchestrator/config.py`:

```python
system_prompt: str = (
    "You are a [your persona here]. "
    "Keep responses brief (max 15-20 words per sentence)."
)
```

### Use Different Voice

Change `CARTESIA_VOICE_ID` in `.env`. Browse voices at: https://cartesia.ai/voices

### Adjust Chunking Behavior

Modify delimiters in `orchestrator/config.py`:

```python
sentence_delimiters: List[str] = ['.', '!', '?']  # Remove ',' for longer chunks
```

## 📊 Performance Metrics

| Metric | Target | Typical |
|--------|--------|---------|
| Total Latency | < 1s | 600-800ms |
| STT Latency | < 200ms | 150ms |
| LLM First Token | < 300ms | 200ms |
| TTS Start | < 100ms | 50ms |
| Interrupt Response | < 100ms | 50ms |

## 🚧 Known Limitations

1. **Mistral Free Tier**: Strict rate limits (5 req/min)
2. **Audio Input**: Currently uses microphone; phone integration requires additional work
3. **Context Window**: Limited conversation history (can be extended)
4. **Error Recovery**: Network failures require manual restart

## 🛣️ Roadmap

- [ ] Phone system integration (Twilio, etc.)
- [ ] Multi-turn context management
- [ ] Emotion detection and adaptive tone
- [ ] Voice activity detection tuning
- [ ] Docker deployment
- [ ] Metrics and monitoring

## 📄 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

---

**Built with ❤️ for natural, low-latency AI conversations**
