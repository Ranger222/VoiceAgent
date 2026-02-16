"""
Local Speech-to-Text using faster-whisper.
Optimized for low-latency turn-taking on M2 Apple Silicon.

Key latency strategy:
  - SHORT silence threshold (300ms) to detect end-of-speech quickly
  - Energy-based VAD with adaptive threshold
  - Immediate transcription when pause detected
"""

import logging
import struct
import time
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class FasterWhisperSTT:
    """
    Low-latency local STT using faster-whisper (CTranslate2).

    Accumulates PCM audio, detects silence via RMS energy,
    and transcribes the buffer as soon as a short pause is detected.
    """

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "auto",
        compute_type: str = "int8",
        sample_rate: int = 16000,
        silence_threshold: float = 0.008,
        silence_duration_ms: int = 300,
        min_speech_ms: int = 250,
    ):
        """
        Args:
            model_size: Whisper model (tiny.en, base.en, small.en)
            device: "auto" lets CTranslate2 pick best backend
            compute_type: "int8" for M2 efficiency
            sample_rate: Expected audio sample rate (16kHz from browser)
            silence_threshold: RMS below this = silence (lowered for sensitivity)
            silence_duration_ms: Silence ms before triggering transcription (key latency knob)
            min_speech_ms: Minimum speech duration to avoid transcribing noise
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration_ms = silence_duration_ms
        self.min_speech_ms = min_speech_ms

        # Model (loaded once at startup)
        self.model: Optional[WhisperModel] = None

        # Audio state
        self._audio_buffer = bytearray()
        self._speech_start_time: Optional[float] = None
        self._last_speech_time: float = 0.0
        self._has_speech = False

        # Adaptive energy tracking
        self._noise_floor = 0.005
        self._noise_samples = 0

    def load_model(self) -> None:
        """Load the Whisper model. Call once at startup."""
        t0 = time.time()
        logger.info(f"Loading faster-whisper model '{self.model_size}' "
                     f"(device={self.device}, compute_type={self.compute_type})...")
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        ms = int((time.time() - t0) * 1000)
        logger.info(f"Whisper model loaded in {ms}ms")

    def feed_audio(self, pcm_bytes: bytes) -> Optional[str]:
        """
        Feed raw PCM s16le audio and return transcript when pause detected.

        Optimized for low latency:
          - Triggers after just 300ms of silence (not 800ms)
          - Skips chunks shorter than min_speech_ms
          - Uses adaptive noise floor

        Args:
            pcm_bytes: Raw 16-bit signed little-endian PCM audio

        Returns:
            Transcript string if end-of-utterance detected, else None
        """
        self._audio_buffer.extend(pcm_bytes)
        rms = self._calculate_rms(pcm_bytes)
        now = time.time()

        # Adaptive noise floor (learn from quiet periods)
        effective_threshold = max(self.silence_threshold, self._noise_floor * 2.5)

        if rms > effective_threshold:
            # Speech detected
            if not self._has_speech:
                self._speech_start_time = now
                self._has_speech = True
            self._last_speech_time = now
            return None
        else:
            # Silence — update noise floor slowly
            if not self._has_speech:
                self._noise_floor = (self._noise_floor * 0.95) + (rms * 0.05)
                self._noise_samples += 1

        # Check if we've had enough silence after speech
        if not self._has_speech:
            # No speech yet, keep only last 0.5s of audio (avoid buffer bloat)
            max_buffer = self.sample_rate * 2 * 0.5  # 0.5s of s16le audio
            if len(self._audio_buffer) > max_buffer:
                self._audio_buffer = self._audio_buffer[-int(max_buffer):]
            return None

        silence_ms = (now - self._last_speech_time) * 1000

        if silence_ms >= self.silence_duration_ms:
            # End of utterance detected — transcribe immediately
            speech_duration_ms = (self._last_speech_time - self._speech_start_time) * 1000

            if speech_duration_ms < self.min_speech_ms:
                # Too short, likely noise — discard
                logger.debug(f"Discarding short audio ({speech_duration_ms:.0f}ms)")
                self._reset_state()
                return None

            audio_data = bytes(self._audio_buffer)
            self._reset_state()

            return self._transcribe(audio_data)

        return None

    def force_transcribe(self) -> Optional[str]:
        """Force transcription of whatever is in the buffer."""
        if len(self._audio_buffer) < self.sample_rate * 2 * 0.3:
            self._reset_state()
            return None

        audio_data = bytes(self._audio_buffer)
        self._reset_state()
        return self._transcribe(audio_data)

    def _transcribe(self, pcm_bytes: bytes) -> Optional[str]:
        """Transcribe raw PCM audio using faster-whisper."""
        if not self.model:
            logger.error("Whisper model not loaded")
            return None

        t0 = time.time()

        # Convert PCM s16le → float32 numpy array
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Transcribe with speed-optimized settings
        segments, info = self.model.transcribe(
            samples,
            beam_size=1,       # Greedy decoding = fastest
            language="en",
            vad_filter=True,   # Let Whisper's VAD clean up
            vad_parameters=dict(
                min_silence_duration_ms=200,
                speech_pad_ms=100,
            ),
        )

        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        text = " ".join(text_parts).strip()

        ms = int((time.time() - t0) * 1000)
        if text:
            logger.info(f"STT ({ms}ms): \"{text}\"")
        else:
            logger.debug(f"STT ({ms}ms): [empty]")

        return text if text else None

    def clear_buffer(self) -> None:
        """Clear all state."""
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset audio buffer and speech tracking state."""
        self._audio_buffer.clear()
        self._has_speech = False
        self._speech_start_time = None
        self._last_speech_time = 0.0

    @staticmethod
    def _calculate_rms(pcm_bytes: bytes) -> float:
        """Calculate RMS energy of PCM s16le audio (normalized 0.0–1.0)."""
        if len(pcm_bytes) < 2:
            return 0.0
        n_samples = len(pcm_bytes) // 2
        samples = np.frombuffer(pcm_bytes[:n_samples * 2], dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(samples * samples)) / 32768.0
        return rms
