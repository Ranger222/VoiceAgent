"""
Main orchestrator for AI Voice Companion.
Coordinates STT, LLM, and TTS with cascaded streaming and barge-in support.
"""

import asyncio
import logging
import signal
import sys
from typing import Optional

from .config import get_config
from .components import FasterWhisperSTT, MistralLLM, PiperTTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VoiceOrchestrator:
    """
    Main orchestrator for the AI Voice Companion.

    Coordinates the flow: Audio → STT → LLM → TTS
    with cascaded streaming and barge-in support.
    """

    def __init__(self):
        """Initialize the orchestrator."""
        # Load configuration
        self.config = get_config()

        # Initialize local STT (faster-whisper)
        self.stt = FasterWhisperSTT(
            model_size=self.config.whisper.model_size,
            device=self.config.whisper.device,
            compute_type=self.config.whisper.compute_type,
            silence_duration_ms=self.config.whisper.silence_duration_ms,
        )

        # Initialize LLM (Mistral — cloud API)
        self.llm = MistralLLM(
            api_key=self.config.mistral.api_key,
            model=self.config.mistral.model,
            system_prompt=self.config.mistral.system_prompt,
            temperature=self.config.mistral.temperature,
            max_tokens=self.config.mistral.max_tokens,
            max_retries=self.config.mistral.max_retries,
            retry_delay_base=self.config.mistral.retry_delay_base,
        )

        # Initialize local TTS (Piper)
        self.tts = PiperTTS(
            voice=self.config.piper.voice,
        )

        # Orchestration state
        self.is_running = False
        self.interrupt_event = asyncio.Event()
        self.shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the orchestrator and load all models."""
        logger.info("Starting AI Voice Companion Orchestrator")

        try:
            loop = asyncio.get_event_loop()

            # Load models (in thread — they're synchronous)
            logger.info("Loading STT model...")
            await loop.run_in_executor(None, self.stt.load_model)

            logger.info("Loading TTS model...")
            await loop.run_in_executor(None, self.tts.load_model)

            self.is_running = True
            logger.info("All models loaded successfully")

            # Start the main loop
            await self._run_loop()

        except Exception as e:
            logger.error(f"Error starting orchestrator: {e}")
            raise
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the orchestrator."""
        logger.info("Stopping AI Voice Companion Orchestrator")
        self.is_running = False
        logger.info("Orchestrator stopped")

    async def _run_loop(self) -> None:
        """
        Main orchestration loop.

        Reads audio from microphone, transcribes locally, streams response,
        and synthesizes speech locally.
        """
        import pyaudio

        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=4096,
        )

        logger.info("🎙️ Listening... (speak into your microphone)")

        try:
            while self.is_running:
                # Read audio chunk
                data = stream.read(4096, exception_on_overflow=False)

                # Feed to local STT
                transcript = self.stt.feed_audio(data)

                if transcript:
                    logger.info(f"👤 User: {transcript}")

                    # Clear interrupt flag
                    self.interrupt_event.clear()

                    # Process through LLM and TTS
                    await self._process_conversation_turn(transcript)

                await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

    async def _process_conversation_turn(self, user_message: str) -> None:
        """Process a single conversation turn with cascaded streaming."""
        try:
            sentence_count = 0

            async for sentence in self.llm.stream_with_chunking(user_message):
                if self.interrupt_event.is_set():
                    logger.info("Conversation turn interrupted")
                    break

                sentence_count += 1
                logger.info(f"🤖 AI (chunk {sentence_count}): {sentence}")

                # Synthesize and play with Piper
                pcm_data = await self.tts.synthesize_async(sentence)

                # Play audio locally
                await self._play_audio(pcm_data)

                if self.interrupt_event.is_set():
                    break

            if sentence_count > 0 and not self.interrupt_event.is_set():
                logger.info(f"✅ Completed response ({sentence_count} chunks)")

        except Exception as e:
            logger.error(f"Error processing conversation turn: {e}")

    async def _play_audio(self, pcm_data: bytes) -> None:
        """Play PCM audio through speakers."""
        import pyaudio

        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.tts.sample_rate,
            output=True,
        )

        try:
            stream.write(pcm_data)
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()


async def main():
    """Main entry point."""
    orchestrator = VoiceOrchestrator()

    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        orchestrator.shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
