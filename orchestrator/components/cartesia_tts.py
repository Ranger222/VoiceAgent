"""
Cartesia Text-to-Speech (TTS) integration.
WebSocket-based streaming audio synthesis.
"""

import asyncio
import logging
from typing import Optional
import json
import base64

import websockets
import pyaudio

logger = logging.getLogger(__name__)


class CartesiaTTS:
    """
    Real-time text-to-speech using Cartesia Sonic WebSocket API.
    
    Handles streaming TTS synthesis and audio playback with interrupt support.
    """
    
    def __init__(
        self,
        api_key: str,
        voice_id: str = "a0e99841-438c-4a64-b679-ae501e7d6091",
        model: str = "sonic-english",
        sample_rate: int = 16000,
    ):
        """
        Initialize Cartesia TTS client.
        
        Args:
            api_key: Cartesia API key
            voice_id: Voice ID to use
            model: Model to use (sonic-english, etc.)
            sample_rate: Audio sample rate
        """
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = model
        self.sample_rate = sample_rate
        
        # WebSocket connection
        self.ws = None
        self.is_connected = False
        
        # Audio playback
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Interrupt handling
        self.is_playing = False
        self.should_stop = asyncio.Event()
        
    async def connect(self) -> None:
        """Establish WebSocket connection to Cartesia."""
        try:
            # Cartesia WebSocket endpoint
            url = f"wss://api.cartesia.ai/tts/websocket?api_key={self.api_key}&cartesia_version=2024-06-10"
            
            self.ws = await websockets.connect(url)
            self.is_connected = True
            
            logger.info("Connected to Cartesia TTS")
            
        except Exception as e:
            logger.error(f"Failed to connect to Cartesia: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        if self.ws:
            await self.ws.close()
            self.is_connected = False
            logger.info("Disconnected from Cartesia")
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        self.audio.terminate()
    
    async def speak(
        self,
        text: str,
        interrupt_event: Optional[asyncio.Event] = None,
    ) -> None:
        """
        Synthesize and play speech from text.
        
        Args:
            text: Text to synthesize
            interrupt_event: Event that signals playback should stop
        """
        if not self.is_connected:
            logger.error("Not connected to Cartesia")
            return
        
        try:
            self.is_playing = True
            self.should_stop.clear()
            
            # Prepare TTS request
            request = {
                "context_id": "voice-companion",
                "model_id": self.model,
                "transcript": text,
                "voice": {
                    "mode": "id",
                    "id": self.voice_id,
                },
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": self.sample_rate,
                },
            }
            
            # Send request
            await self.ws.send(json.dumps(request))
            logger.debug(f"Sent TTS request: {text}")
            
            # Initialize audio stream if needed
            if not self.stream:
                self.stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.sample_rate,
                    output=True,
                )
            
            # Receive and play audio chunks
            while True:
                # Check for interrupt
                if interrupt_event and interrupt_event.is_set():
                    logger.info("Playback interrupted")
                    break
                
                if self.should_stop.is_set():
                    logger.info("Playback stopped")
                    break
                
                try:
                    # Receive audio chunk with timeout
                    response = await asyncio.wait_for(
                        self.ws.recv(),
                        timeout=0.1
                    )
                    
                    data = json.loads(response)
                    
                    # Check for audio data
                    if "data" in data:
                        # Decode base64 audio
                        audio_data = base64.b64decode(data["data"])
                        
                        # Play audio
                        self.stream.write(audio_data)
                    
                    # Check if done
                    if data.get("done", False):
                        logger.debug("TTS synthesis complete")
                        break
                        
                except asyncio.TimeoutError:
                    # No data yet, continue
                    continue
                except Exception as e:
                    logger.error(f"Error receiving audio: {e}")
                    break
            
        except Exception as e:
            logger.error(f"Error in TTS synthesis: {e}")
        
        finally:
            self.is_playing = False
    
    async def cancel_playback(self) -> None:
        """Immediately stop current audio playback."""
        self.should_stop.set()
        
        # Clear audio buffer
        if self.stream:
            try:
                # Stop and restart stream to clear buffer
                self.stream.stop_stream()
                self.stream.close()
                
                self.stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.sample_rate,
                    output=True,
                )
            except Exception as e:
                logger.error(f"Error clearing audio buffer: {e}")
        
        logger.info("Playback cancelled")
    
    def is_currently_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self.is_playing


class TTSFallback:
    """
    Fallback TTS for testing without Cartesia API.
    Just logs the text instead of synthesizing speech.
    """
    
    def __init__(self):
        self.is_connected = True
        self.is_playing = False
    
    async def connect(self) -> None:
        """Mock connection."""
        logger.info("Using TTS fallback (text-only mode)")
    
    async def disconnect(self) -> None:
        """Mock disconnection."""
        pass
    
    async def speak(
        self,
        text: str,
        interrupt_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Log text instead of speaking."""
        self.is_playing = True
        logger.info(f"🔊 TTS: {text}")
        
        # Simulate speech duration (roughly 150 words per minute)
        words = len(text.split())
        duration = (words / 150) * 60  # seconds
        
        try:
            await asyncio.sleep(duration)
        except asyncio.CancelledError:
            logger.info("TTS interrupted")
        finally:
            self.is_playing = False
    
    async def cancel_playback(self) -> None:
        """Mock cancel."""
        self.is_playing = False
    
    def is_currently_playing(self) -> bool:
        """Check if playing."""
        return self.is_playing
