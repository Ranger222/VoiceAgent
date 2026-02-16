"""
Test the full AI Voice Companion integration.
This tests Mistral LLM streaming with real API.
"""

import asyncio
import sys
sys.path.insert(0, '.')

from orchestrator.config import get_config
from orchestrator.components.mistral_llm import MistralLLM


async def test_mistral_streaming():
    """Test Mistral LLM with streaming and chunking."""
    print("\n" + "=" * 60)
    print("Testing Mistral LLM Streaming Integration")
    print("=" * 60)
    
    config = get_config()
    
    llm = MistralLLM(
        api_key=config.mistral.api_key,
        model=config.mistral.model,
        temperature=config.mistral.temperature,
        max_tokens=config.mistral.max_tokens,
    )
    
    test_message = "Hello! Tell me a very short joke."
    
    print(f"\n👤 User: {test_message}")
    print(f"🤖 AI: ", end="", flush=True)
    
    full_response = ""
    chunk_count = 0
    
    try:
        async for sentence in llm.stream_with_chunking(test_message):
            chunk_count += 1
            full_response += sentence + " "
            print(f"\n  [Chunk {chunk_count}]: {sentence}", flush=True)
        
        print(f"\n\n✅ Streaming complete!")
        print(f"✅ Total chunks: {chunk_count}")
        print(f"✅ Full response: {full_response.strip()}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    print("\n🚀 AI Voice Companion - Mistral Integration Test\n")
    
    asyncio.run(test_mistral_streaming())
    
    print("\n✅ Mistral integration working!")
    print("\nThe full orchestrator is ready to run.")
    print("Run: python -m orchestrator.main")
    print()
