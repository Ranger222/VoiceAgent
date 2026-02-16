"""
Sentence chunking logic for cascaded streaming.
Buffers LLM tokens and yields complete sentences for TTS.
"""

import re
from typing import AsyncIterator, List


class SentenceChunker:
    """
    Intelligent sentence boundary detection for streaming LLM responses.
    
    Buffers tokens until a sentence boundary is detected, then yields
    the complete sentence for TTS processing.
    """
    
    def __init__(
        self,
        delimiters: List[str] = None,
        min_length: int = 10,
        max_length: int = 200,
    ):
        """
        Initialize the sentence chunker.
        
        Args:
            delimiters: Punctuation marks that indicate sentence boundaries
            min_length: Minimum characters before considering a chunk complete
            max_length: Maximum characters before forcing a chunk split
        """
        self.delimiters = delimiters or ['.', '!', '?', ',']
        self.min_length = min_length
        self.max_length = max_length
        self.buffer = ""
        
        # Common abbreviations that shouldn't trigger sentence breaks
        self.abbreviations = {
            'dr.', 'mr.', 'mrs.', 'ms.', 'prof.', 'sr.', 'jr.',
            'etc.', 'vs.', 'e.g.', 'i.e.', 'a.m.', 'p.m.',
        }
    
    async def process_stream(self, token_stream: AsyncIterator[str]) -> AsyncIterator[str]:
        """
        Process a stream of tokens and yield complete sentences.
        
        Args:
            token_stream: Async iterator of text tokens from LLM
            
        Yields:
            Complete sentences ready for TTS
        """
        async for token in token_stream:
            self.buffer += token
            
            # Check if we've hit max length - force a split
            if len(self.buffer) >= self.max_length:
                chunk = self._force_split()
                if chunk:
                    yield chunk
                    continue
            
            # Check for sentence boundaries
            if self._has_sentence_boundary():
                chunk = self._extract_sentence()
                if chunk:
                    yield chunk
        
        # Yield any remaining buffer at the end
        if self.buffer.strip():
            yield self.buffer.strip()
            self.buffer = ""
    
    def _has_sentence_boundary(self) -> bool:
        """Check if the buffer contains a sentence boundary."""
        if len(self.buffer) < self.min_length:
            return False
        
        # Check for any delimiter
        for delimiter in self.delimiters:
            if delimiter in self.buffer:
                # Make sure it's not an abbreviation
                if not self._is_abbreviation():
                    return True
        
        return False
    
    def _is_abbreviation(self) -> bool:
        """Check if the last period is part of an abbreviation."""
        # Get the last few words
        words = self.buffer.lower().split()
        if not words:
            return False
        
        last_word = words[-1]
        
        # Check against known abbreviations
        return last_word in self.abbreviations
    
    def _extract_sentence(self) -> str:
        """Extract the first complete sentence from the buffer."""
        # Find the position of the first delimiter after min_length
        for i in range(self.min_length, len(self.buffer)):
            if self.buffer[i] in self.delimiters:
                # Check if it's an abbreviation
                potential_sentence = self.buffer[:i+1]
                words = potential_sentence.lower().split()
                
                if words and words[-1] not in self.abbreviations:
                    # Extract the sentence
                    sentence = self.buffer[:i+1].strip()
                    self.buffer = self.buffer[i+1:].lstrip()
                    return sentence
        
        return ""
    
    def _force_split(self) -> str:
        """Force a split at max_length, preferring word boundaries."""
        # Try to split at a word boundary
        split_pos = self.max_length
        
        # Look backwards for a space
        for i in range(self.max_length, max(0, self.max_length - 50), -1):
            if self.buffer[i] == ' ':
                split_pos = i
                break
        
        chunk = self.buffer[:split_pos].strip()
        self.buffer = self.buffer[split_pos:].lstrip()
        return chunk
    
    def reset(self):
        """Clear the buffer."""
        self.buffer = ""


def chunk_text_sync(text: str, delimiters: List[str] = None) -> List[str]:
    """
    Synchronous helper to chunk a complete text into sentences.
    Useful for testing.
    
    Args:
        text: Complete text to chunk
        delimiters: Sentence delimiters
        
    Returns:
        List of sentence chunks
    """
    delimiters = delimiters or ['.', '!', '?']
    
    # Simple regex-based splitting for complete text
    pattern = '|'.join(re.escape(d) for d in delimiters)
    chunks = re.split(f'({pattern})', text)
    
    # Recombine chunks with their delimiters
    result = []
    for i in range(0, len(chunks) - 1, 2):
        sentence = (chunks[i] + chunks[i + 1]).strip()
        if sentence:
            result.append(sentence)
    
    # Add any remaining text
    if len(chunks) % 2 == 1 and chunks[-1].strip():
        result.append(chunks[-1].strip())
    
    return result
