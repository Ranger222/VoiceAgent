"""
Deepgram Speech-to-Text (STT) integration.
WebSocket-based real-time transcription with Voice Activity Detection.
Updated for Deepgram SDK v5.x
"""

import asyncio
import logging
from typing import AsyncIterator, Optional
import json

from deepgram import DeepgramClient

logger = logging.getLogger(__name__)


class DeepgramSTT:
    """
    Real-time speech-to-text using Deepgram WebSocket API.
    
    Handles audio streaming, transcription, and interrupt detection.
    Compatible with Deepgram SDK v5.x
    """
    
    def __init__(self, api_key: str, model: str = "nova-2", vad_threshold_ms: int = 400):
        """
        Initialize Deepgram STT client.
        
        Args:
            api_key: Deepgram API key
            model: Model to use (nova-2, nova-3, etc.)
            vad_threshold_ms: Voice Activity Detection threshold in milliseconds
        """
        self.api_key = api_key
        self.model = model
        self.vad_threshold_ms = vad_threshold_ms
        
        # Client configuration
        self.client = DeepgramClient(api_key=api_key)
        
        # Connection state
        self.connection = None
        self.is_listening = False
        self.transcript_queue = asyncio.Queue()
        self.interrupt_detected = asyncio.Event()
        
    async def connect(self) -> None:
        """Establish WebSocket connection to Deepgram."""
        try:
            # Get WebSocket connection using v5.x API
            # The connect() method returns an iterator with the socket client
            connection_iter = self.client.listen.v1.connect(
                model=self.model,
                language="en-US",
                encoding="linear16",
                sample_rate="16000",
                channels="1",
                interim_results="true",
                punctuate="true",
                smart_format="true",
                vad_events="true",
                endpointing=str(self.vad_threshold_ms),
            )
            
            # Get the socket client from the iterator
            self.connection = next(connection_iter)
            
            # Set up event handlers
            self.connection.on_message = self._on_message
            self.connection.on_error = self._on_error
            self.connection.on_close = self._on_close
            
            # Start the connection
            await self.connection.start()
            
            self.is_listening = True
            logger.info(f"Connected to Deepgram with model: {self.model}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Deepgram: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        if self.connection:
            await self.connection.finish()
            self.is_listening = False
            logger.info("Disconnected from Deepgram")
    
    def _on_message(self, message: dict) -> None:
        """Handle messages from Deepgram."""
        try:
            # Check if this is a transcript message
            if message.get("type") == "Results":
                channel = message.get("channel", {})
                alternatives = channel.get("alternatives", [])
                
                if alternatives:
                    transcript = alternatives[0].get("transcript", "")
                    is_final = message.get("is_final", False)
                    
                    if is_final and transcript.strip():
                        logger.debug(f"Transcript: {transcript}")
                        
                        # Put transcript in queue for processing
                        asyncio.create_task(self.transcript_queue.put(transcript))
                        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _on_error(self, error: Exception) -> None:
        """Handle error events from Deepgram."""
        logger.error(f"Deepgram error: {error}")
    
    def _on_close(self) -> None:
        """Handle connection close."""
        logger.info("Deepgram connection closed")
        self.is_listening = False
    
    async def send_audio(self, audio_data: bytes) -> None:
        """
        Send audio data to Deepgram for transcription.
        
        Args:
            audio_data: Raw audio bytes (PCM 16-bit, 16kHz)
        """
        if self.connection and self.is_listening:
            await self.connection.send(audio_data)
    
    async def stream_transcripts(self) -> AsyncIterator[str]:
        """
        Stream transcribed text as it becomes available.
        
        Yields:
            Transcribed text segments
        """
        while self.is_listening:
            try:
                # Wait for transcript with timeout
                transcript = await asyncio.wait_for(
                    self.transcript_queue.get(),
                    timeout=0.1
                )
                yield transcript
                
            except asyncio.TimeoutError:
                # No transcript available, continue
                continue
            except Exception as e:
                logger.error(f"Error streaming transcript: {e}")
                break
    
    async def detect_interruption(self, timeout: float = 0.1) -> bool:
        """
        Check if user has started speaking (for barge-in detection).
        
        Args:
            timeout: How long to wait for speech detection
            
        Returns:
            True if user speech detected, False otherwise
        """
        try:
            # Check if there's a new transcript (indicates user is speaking)
            transcript = await asyncio.wait_for(
                self.transcript_queue.get(),
                timeout=timeout
            )
            
            if transcript.strip():
                # Put it back for processing
                await self.transcript_queue.put(transcript)
                return True
                
        except asyncio.TimeoutError:
            return False
        
        return False
    
    def clear_queue(self) -> None:
        """Clear the transcript queue (useful after interruption)."""
        while not self.transcript_queue.empty():
            try:
                self.transcript_queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class AudioStreamSimulator:
    """
    Simulates an audio stream for testing purposes.
    In production, replace with actual microphone input.
    """
    
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 4096):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
    
    async def stream_audio(self) -> AsyncIterator[bytes]:
        """
        Simulate audio streaming.
        
        In production, this would read from a microphone using PyAudio.
        """
        import pyaudio
        
        audio = pyaudio.PyAudio()
        
        try:
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )
            
            logger.info("Audio stream started")
            
            while True:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                yield data
                await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning
                
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            logger.info("Audio stream stopped")
