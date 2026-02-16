"""
Mistral LLM integration with streaming support.
Handles rate limiting, retries, and sentence chunking.
"""

import asyncio
import logging
from typing import AsyncIterator, List, Optional
import time

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    logger.warning("Mistral SDK not available")

from .chunker import SentenceChunker

logger = logging.getLogger(__name__)


class MistralLLM:
    """
    Streaming LLM client for Mistral API with cascaded output.
    
    Implements rate limit handling and sentence chunking for low-latency TTS.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "mistral-small-latest",
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 150,
        max_retries: int = 3,
        retry_delay_base: float = 1.0,
    ):
        """
        Initialize Mistral LLM client.
        
        Args:
            api_key: Mistral API key
            model: Model to use
            system_prompt: System prompt for AI persona
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            max_retries: Maximum retry attempts for rate limits
            retry_delay_base: Base delay for exponential backoff
        """
        if not MISTRAL_AVAILABLE:
            raise Exception("Mistral SDK not available")
        
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay_base = retry_delay_base
        
        # System prompt for conversational AI
        self.system_prompt = system_prompt or (
            "You are a warm, empathetic, and unfiltered AI companion. "
            "You are talking to a human over a phone-like interface. "
            "Be casual, use natural fillers like 'mhm' or 'I hear you,' "
            "and never lecture the user. Keep your responses brief "
            "(max 15-20 words per sentence) to ensure low latency. "
            "Use short, conversational sentences."
        )
        
        # Conversation history (using dict format for v1.x SDK)
        self.conversation_history: List[dict] = []
        
        # Chunker for sentence-level streaming
        self.chunker = SentenceChunker()
        
    async def stream_response(self, user_message: str) -> AsyncIterator[str]:
        """
        Stream LLM response tokens.
        
        Args:
            user_message: User's input text
            
        Yields:
            Text tokens from the LLM
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Prepare messages with system prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history,
        ]
        
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                # Stream response from Mistral
                response_stream = self.client.chat.stream(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                # Collect full response for history
                full_response = ""
                
                # Stream tokens
                for chunk in response_stream:
                    if chunk.data.choices:
                        delta = chunk.data.choices[0].delta.content
                        if delta:
                            full_response += delta
                            yield delta
                
                # Add assistant response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": full_response
                })
                
                logger.info(f"LLM response: {full_response}")
                break  # Success, exit retry loop
                
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a rate limit error
                if "429" in error_str or "rate limit" in error_str.lower():
                    retry_count += 1
                    
                    if retry_count > self.max_retries:
                        logger.error("Max retries exceeded for rate limit")
                        yield "I need a moment to think. Please try again in a few seconds."
                        break
                    
                    # Exponential backoff
                    delay = self.retry_delay_base * (2 ** (retry_count - 1))
                    logger.warning(f"Rate limited. Retrying in {delay}s (attempt {retry_count}/{self.max_retries})")
                    await asyncio.sleep(delay)
                    
                else:
                    # Non-rate-limit error
                    logger.error(f"Error streaming from Mistral: {e}")
                    yield "I'm having trouble processing that. Could you try again?"
                    break
    
    async def stream_with_chunking(self, user_message: str) -> AsyncIterator[str]:
        """
        Stream LLM response with sentence-level chunking.
        
        This is the key method for cascaded streaming - it yields complete
        sentences as soon as they're available, rather than waiting for the
        full response.
        
        Args:
            user_message: User's input text
            
        Yields:
            Complete sentences ready for TTS
        """
        # Reset chunker buffer
        self.chunker.reset()
        
        # Process the token stream through the chunker
        async for sentence in self.chunker.process_stream(
            self.stream_response(user_message)
        ):
            logger.debug(f"Sentence chunk: {sentence}")
            yield sentence
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_history(self) -> List[dict]:
        """Get conversation history."""
        return self.conversation_history.copy()


# Async wrapper for synchronous Mistral client
async def async_stream_wrapper(sync_generator):
    """
    Wrap a synchronous generator to make it async.
    
    This is needed because Mistral's client is synchronous but we need
    async for proper orchestration.
    """
    loop = asyncio.get_event_loop()
    
    def get_next():
        try:
            return next(sync_generator)
        except StopIteration:
            return None
    
    while True:
        chunk = await loop.run_in_executor(None, get_next)
        if chunk is None:
            break
        yield chunk
