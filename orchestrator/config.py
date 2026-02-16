"""
Configuration management for AI Voice Companion Orchestrator.
Handles environment variables, API keys, and system settings.
"""

import os
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class WhisperConfig:
    """Configuration for local faster-whisper STT."""
    model_size: str = "base.en"
    device: str = "auto"
    compute_type: str = "int8"
    language: str = "en"
    sample_rate: int = 16000
    silence_threshold: float = 0.01
    silence_duration_ms: int = 800


@dataclass
class MistralConfig:
    """Configuration for Mistral LLM service."""
    api_key: str
    model: str = "mistral-small-latest"
    temperature: float = 0.7
    max_tokens: int = 150  # Keep responses concise
    stream: bool = True

    # Rate limiting
    max_retries: int = 3
    retry_delay_base: float = 1.0  # Base delay for exponential backoff

    # System prompt for conversational AI
    system_prompt: str = (
        "You are a warm, empathetic, and unfiltered AI companion. "
        "You are talking to a human over a phone-like interface. "
        "Be casual, use natural fillers like 'mhm' or 'I hear you,' "
        "and never lecture the user. Keep your responses brief "
        "(max 15-20 words per sentence) to ensure low latency. "
        "Use short, conversational sentences."
    )


@dataclass
class PiperConfig:
    """Configuration for local Piper TTS."""
    voice: str = "en_US-amy-medium"
    sample_rate: int = 22050  # Piper default


@dataclass
class ChunkerConfig:
    """Configuration for sentence chunking logic."""
    # Punctuation marks that trigger chunk completion
    sentence_delimiters: List[str] = None

    # Minimum chunk length to avoid micro-chunks
    min_chunk_length: int = 10

    # Maximum chunk length before forcing a split
    max_chunk_length: int = 200

    def __post_init__(self):
        if self.sentence_delimiters is None:
            self.sentence_delimiters = ['.', '!', '?', ',']


@dataclass
class OrchestratorConfig:
    """Main configuration for the orchestrator."""
    whisper: WhisperConfig
    mistral: MistralConfig
    piper: PiperConfig
    chunker: ChunkerConfig

    # Audio settings
    audio_chunk_size: int = 4096

    # Barge-in settings
    interrupt_detection_enabled: bool = True
    interrupt_cooldown_ms: int = 100  # Minimum time between interrupts

    # Logging
    log_level: str = "INFO"
    log_transcripts: bool = True
    log_llm_responses: bool = True


def load_config() -> OrchestratorConfig:
    """
    Load configuration from environment variables.

    Returns:
        OrchestratorConfig: Fully configured orchestrator settings

    Raises:
        ValueError: If required API keys are missing
    """
    # Only Mistral API key is required (STT/TTS are local)
    mistral_key = os.getenv("MISTRAL_API_KEY")

    if not mistral_key:
        raise ValueError("MISTRAL_API_KEY environment variable is required")

    # Build configuration
    config = OrchestratorConfig(
        whisper=WhisperConfig(
            model_size=os.getenv("WHISPER_MODEL", "base.en"),
            device=os.getenv("WHISPER_DEVICE", "auto"),
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            silence_duration_ms=int(os.getenv("VAD_THRESHOLD_MS", "800")),
        ),
        mistral=MistralConfig(
            api_key=mistral_key,
            model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
            temperature=float(os.getenv("MISTRAL_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MISTRAL_MAX_TOKENS", "150")),
        ),
        piper=PiperConfig(
            voice=os.getenv("PIPER_VOICE", "en_US-amy-medium"),
        ),
        chunker=ChunkerConfig(
            min_chunk_length=int(os.getenv("MIN_CHUNK_LENGTH", "10")),
            max_chunk_length=int(os.getenv("MAX_CHUNK_LENGTH", "200")),
        ),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )

    return config


# Global config instance (lazy loaded)
_config: OrchestratorConfig = None


def get_config() -> OrchestratorConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
