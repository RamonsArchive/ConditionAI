#!/usr/bin/env python3
"""
Simple test script for Llama 3 8B model
"""

from gpt4all import GPT4All

def test_llama_model():
    """Test the Llama model with sample titles"""
    
    print("🤖 Loading Llama 3 8B model...")
    model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")
    print("✅ Model loaded successfully!")
    
    # Test titles from your updated test_items.csv
    test_titles = [
        "Vintage Bicycle",
        "Count for sale for free",
        "Used couch for sale asap", 
        "free sofa pickup only",
        "lighlty used couch for heavy discount",
        "free couch",
        "used funiture need gone asap"
    ]
    
    print("\n🧪 Testing object detection...")
    
    with model.chat_session():
        for title in test_titles:
            print(f"\n--- Testing: '{title}' ---")
            prompt = f"Give generic type of this item as one word or tag: {title}"
            response = model.generate(prompt, max_tokens=10)
            print(f"Response: {response.strip()}")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_llama_model()
