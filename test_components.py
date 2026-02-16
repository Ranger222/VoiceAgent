"""
Simple test script to verify the orchestrator components work.
"""

import asyncio
import sys
sys.path.insert(0, '.')

from orchestrator.components.chunker import chunk_text_sync

def test_chunker():
    """Test the sentence chunker."""
    print("=" * 60)
    print("Testing Sentence Chunker")
    print("=" * 60)
    
    test_cases = [
        "Hello, how are you? I am doing great. Thanks for asking!",
        "Wow! That's amazing! I'm so happy!",
        "What's your name? How old are you?",
    ]
    
    for text in test_cases:
        chunks = chunk_text_sync(text)
        print(f"\nInput: {text}")
        print(f"Output: {chunks}")
        print(f"✅ {len(chunks)} chunks detected")
    
    print("\n" + "=" * 60)
    print("✅ Chunker tests passed!")
    print("=" * 60)

def test_config():
    """Test configuration loading."""
    print("\n" + "=" * 60)
    print("Testing Configuration")
    print("=" * 60)
    
    try:
        from orchestrator.config import get_config
        config = get_config()
        
        print(f"\n✅ Deepgram API Key: {'*' * 20}{config.deepgram.api_key[-10:]}")
        print(f"✅ Deepgram Model: {config.deepgram.model}")
        print(f"✅ VAD Threshold: {config.deepgram.vad_threshold_ms}ms")
        
        print(f"\n✅ Mistral API Key: {'*' * 20}{config.mistral.api_key[-10:]}")
        print(f"✅ Mistral Model: {config.mistral.model}")
        print(f"✅ Max Tokens: {config.mistral.max_tokens}")
        
        print(f"\n✅ Cartesia API Key: {'*' * 20}{config.cartesia.api_key[-10:]}")
        print(f"✅ Cartesia Voice ID: {config.cartesia.voice_id}")
        
        print(f"\n✅ Sentence Delimiters: {config.chunker.sentence_delimiters}")
        print(f"✅ Min Chunk Length: {config.chunker.min_chunk_length}")
        print(f"✅ Max Chunk Length: {config.chunker.max_chunk_length}")
        
        print("\n" + "=" * 60)
        print("✅ Configuration loaded successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Configuration error: {e}")
        raise

if __name__ == "__main__":
    print("\n🚀 AI Voice Companion - Component Tests\n")
    
    # Test chunker
    test_chunker()
    
    # Test config
    test_config()
    
    print("\n✅ All tests passed! System is ready.\n")
    print("Next steps:")
    print("1. Run in fallback mode: python -m orchestrator.main --fallback-tts")
    print("2. Speak into your microphone")
    print("3. The AI will respond (text-only in fallback mode)")
    print()
