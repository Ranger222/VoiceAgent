"""
Unit tests for the sentence chunker.
"""

import pytest
import asyncio
from orchestrator.components.chunker import SentenceChunker, chunk_text_sync


class TestSentenceChunker:
    """Test cases for sentence chunking logic."""
    
    def test_chunk_text_sync_basic(self):
        """Test basic sentence splitting."""
        text = "Hello, how are you? I am doing great. Thanks for asking!"
        chunks = chunk_text_sync(text)
        
        assert len(chunks) == 4
        assert chunks[0] == "Hello,"
        assert chunks[1] == "how are you?"
        assert chunks[2] == "I am doing great."
        assert chunks[3] == "Thanks for asking!"
    
    def test_chunk_text_sync_exclamation(self):
        """Test exclamation marks."""
        text = "Wow! That's amazing! I'm so happy!"
        chunks = chunk_text_sync(text)
        
        assert len(chunks) == 3
        assert all(chunk.endswith('!') for chunk in chunks)
    
    def test_chunk_text_sync_question(self):
        """Test question marks."""
        text = "What's your name? How old are you? Where are you from?"
        chunks = chunk_text_sync(text)
        
        assert len(chunks) == 3
        assert all(chunk.endswith('?') for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_chunker_stream_basic(self):
        """Test streaming chunker with basic input."""
        chunker = SentenceChunker(min_length=5)
        
        # Simulate token stream
        async def token_stream():
            tokens = ["Hello", ", ", "how ", "are ", "you", "? ", "I'm ", "good", "."]
            for token in tokens:
                yield token
        
        chunks = []
        async for chunk in chunker.process_stream(token_stream()):
            chunks.append(chunk)
        
        assert len(chunks) == 2
        assert "Hello, how are you?" in chunks[0]
        assert "I'm good." in chunks[1]
    
    @pytest.mark.asyncio
    async def test_chunker_abbreviations(self):
        """Test that abbreviations don't trigger false splits."""
        chunker = SentenceChunker(min_length=5)
        
        async def token_stream():
            tokens = ["Dr", ". ", "Smith ", "is ", "here", "."]
            for token in tokens:
                yield token
        
        chunks = []
        async for chunk in chunker.process_stream(token_stream()):
            chunks.append(chunk)
        
        # Should be one chunk (abbreviation shouldn't split)
        assert len(chunks) == 1
        assert "Dr. Smith is here." in chunks[0]
    
    @pytest.mark.asyncio
    async def test_chunker_max_length(self):
        """Test force split at max length."""
        chunker = SentenceChunker(min_length=5, max_length=30)
        
        async def token_stream():
            # Long sentence without punctuation
            text = "This is a very long sentence that exceeds the maximum length"
            for char in text:
                yield char
        
        chunks = []
        async for chunk in chunker.process_stream(token_stream()):
            chunks.append(chunk)
        
        # Should be split due to max length
        assert len(chunks) > 1
        assert all(len(chunk) <= 35 for chunk in chunks)  # Allow some margin
    
    def test_chunker_reset(self):
        """Test buffer reset."""
        chunker = SentenceChunker()
        chunker.buffer = "Some text"
        
        chunker.reset()
        
        assert chunker.buffer == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
