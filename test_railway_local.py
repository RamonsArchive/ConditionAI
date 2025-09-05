#!/usr/bin/env python3
"""
Test script to verify railway_main.py works locally
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work"""
    try:
        print("🔍 Testing imports...")
        
        # Test basic imports
        import torch
        print("✅ torch imported")
        
        import clip
        print("✅ clip imported")
        
        import pandas as pd
        print("✅ pandas imported")
        
        from litserve import LitAPI, LitServer
        print("✅ litserve imported")
        
        from pydantic import BaseModel
        print("✅ pydantic imported")
        
        print("✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_csv_processing():
    """Test CSV processing logic"""
    try:
        print("\n🔍 Testing CSV processing...")
        
        import pandas as pd
        
        # Check if CSV exists
        if not os.path.exists("listings_from_txt.csv"):
            print("❌ listings_from_txt.csv not found")
            return False
        
        # Read CSV
        df = pd.read_csv("listings_from_txt.csv")
        print(f"✅ CSV loaded: {len(df)} items")
        
        # Test first few rows
        print("📋 Sample data:")
        for i, row in df.head(3).iterrows():
            print(f"  {i+1}. {row['title']} - {row['price']}")
        
        return True
        
    except Exception as e:
        print(f"❌ CSV processing error: {e}")
        return False

def test_model_loading():
    """Test CLIP model loading (without GPU)"""
    try:
        print("\n🔍 Testing CLIP model loading...")
        
        import torch
        import clip
        
        device = "cpu"  # Use CPU for testing
        print(f"Using device: {device}")
        
        print("Loading CLIP model...")
        model, preprocess = clip.load("ViT-L/14", device=device)
        print("✅ CLIP model loaded successfully!")
        
        # Test preprocessing
        from PIL import Image
        import numpy as np
        
        # Create a dummy image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        processed = preprocess(dummy_image)
        print(f"✅ Image preprocessing works: {processed.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Railway Main Components")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("CSV Processing", test_csv_processing),
        ("Model Loading", test_model_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 All tests passed! Ready for deployment.")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
    
    return all_passed

if __name__ == "__main__":
    main()
