"""
Local Text-to-Speech using Piper TTS.
Runs entirely on-device using ONNX runtime.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Piper voice models are downloaded to this cache directory
PIPER_CACHE_DIR = Path.home() / ".local" / "share" / "piper" / "voices"


class PiperTTS:
    """
    Local text-to-speech using Piper (ONNX-based neural TTS).

    Synthesizes speech entirely on-device. No API keys needed.
    Output: 16-bit PCM audio at 22050 Hz.
    """

    def __init__(
        self,
        voice: str = "en_US-amy-medium",
        sample_rate: int = 22050,
    ):
        """
        Args:
            voice: Piper voice model name (auto-downloaded if missing)
            sample_rate: Output sample rate (Piper default is 22050)
        """
        self.voice_name = voice
        self.sample_rate = sample_rate
        self._voice = None

    def load_model(self) -> None:
        """Load the Piper voice model. Call once at startup."""
        from piper import PiperVoice

        t0 = time.time()
        logger.info(f"Loading Piper voice model '{self.voice_name}'...")

        # Determine model path
        model_path = self._get_model_path()

        if model_path is None:
            logger.info(f"Downloading Piper voice model '{self.voice_name}'...")
            model_path = self._download_model()

        self._voice = PiperVoice.load(str(model_path))
        self.sample_rate = self._voice.config.sample_rate

        ms = int((time.time() - t0) * 1000)
        logger.info(f"Piper voice loaded in {ms}ms (sample_rate={self.sample_rate})")

    def synthesize(self, text: str) -> bytes:
        """
        Synthesize speech from text.

        Args:
            text: Text to speak

        Returns:
            Raw PCM s16le audio bytes at self.sample_rate
        """
        import numpy as np

        if not self._voice:
            raise RuntimeError("Piper voice not loaded. Call load_model() first.")

        t0 = time.time()

        # Piper 1.4 synthesize() yields AudioChunk objects with float32 arrays
        pcm_parts = []
        for audio_chunk in self._voice.synthesize(text):
            # Convert float32 [-1.0, 1.0] → int16 PCM
            audio_f32 = audio_chunk.audio_float_array
            audio_i16 = (audio_f32 * 32767).astype(np.int16)
            pcm_parts.append(audio_i16.tobytes())

        pcm_data = b"".join(pcm_parts)

        ms = int((time.time() - t0) * 1000)
        duration_s = len(pcm_data) / (self.sample_rate * 2)
        logger.info(f"TTS ({ms}ms): \"{text[:60]}\" → {duration_s:.1f}s audio")

        return pcm_data

    async def synthesize_async(self, text: str) -> bytes:
        """Async wrapper — offloads synthesis to a thread."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.synthesize, text)

    def _get_model_path(self) -> Optional[Path]:
        """Find existing model file in cache."""
        # Check common locations
        search_dirs = [
            PIPER_CACHE_DIR,
            Path.home() / ".cache" / "piper",
            Path(__file__).parent.parent.parent / "models" / "piper",
        ]

        for d in search_dirs:
            onnx_path = d / f"{self.voice_name}.onnx"
            if onnx_path.exists():
                return onnx_path

        return None

    def _download_model(self) -> Path:
        """Download the voice model from Hugging Face."""
        try:
            from huggingface_hub import hf_hub_download

            # Piper voices are hosted on HuggingFace
            repo_id = "rhasspy/piper-voices"

            # Build the path within the repo
            # Voice name format: lang_REGION-name-quality
            parts = self.voice_name.split("-")
            lang_region = parts[0]  # e.g., en_US
            lang = lang_region.split("_")[0]  # e.g., en
            name = parts[1] if len(parts) > 1 else "amy"
            quality = parts[2] if len(parts) > 2 else "medium"

            # Path in the HF repo
            hf_path = f"{lang}/{lang_region}/{name}/{quality}/{self.voice_name}.onnx"
            json_path = f"{lang}/{lang_region}/{name}/{quality}/{self.voice_name}.onnx.json"

            # Download model and config
            cache_dir = PIPER_CACHE_DIR
            cache_dir.mkdir(parents=True, exist_ok=True)

            model_file = hf_hub_download(
                repo_id=repo_id,
                filename=hf_path,
                cache_dir=str(cache_dir),
                local_dir=str(cache_dir),
            )
            hf_hub_download(
                repo_id=repo_id,
                filename=json_path,
                cache_dir=str(cache_dir),
                local_dir=str(cache_dir),
            )

            # The files are downloaded to a nested path, find the onnx file
            model_path = Path(model_file)
            if not model_path.exists():
                # Try the local_dir path
                model_path = cache_dir / hf_path

            logger.info(f"Downloaded Piper model to {model_path}")
            return model_path

        except Exception as e:
            logger.error(f"Failed to download Piper model: {e}")
            raise
