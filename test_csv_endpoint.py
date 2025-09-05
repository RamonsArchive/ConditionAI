#!/usr/bin/env python3
"""
Test the CSV processing endpoint locally
"""

import requests
import json
import time

def test_csv_endpoint():
    """Test the /process-csv endpoint"""
    try:
        print("🚀 Starting test server...")
        print("   (Make sure railway_main.py is running in another terminal)")
        print("   Run: python railway_main.py")
        print()
        
        # Wait for user to start server
        input("Press Enter when the server is running...")
        
        print("🔍 Testing /process-csv endpoint...")
        
        # Test the CSV processing endpoint
        response = requests.post(
            "http://localhost:8000/process-csv",
            timeout=60  # 60 second timeout for processing
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ CSV processing successful!")
            print(f"📊 Processed {result.get('total_processed', 0)} items")
            print(f"⏱️  Processing time: {result.get('processing_time', 0):.2f}s")
            
            # Show sample results
            results = result.get('results', [])
            if results:
                print("\n📋 Sample results:")
                for i, item in enumerate(results[:3]):
                    print(f"  {i+1}. {item.get('title', 'N/A')}")
                    print(f"     Object: {item.get('detected_object', 'N/A')}")
                    print(f"     Condition: {item.get('condition', 'N/A')}")
                    print(f"     Confidence: {item.get('condition_confidence', 0):.3f}")
                    print()
            
            return True
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Is the server running?")
        print("   Run: python railway_main.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_health_endpoint():
    """Test the health endpoint"""
    try:
        print("🔍 Testing /health endpoint...")
        
        response = requests.get("http://localhost:8000/health", timeout=10)
        
        if response.status_code == 200:
            health = response.json()
            print("✅ Health check successful!")
            print(f"   Status: {health.get('status', 'unknown')}")
            print(f"   Device: {health.get('device', 'unknown')}")
            print(f"   Ollama: {health.get('ollama_available', False)}")
            return True
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing CSV Processing Endpoint")
    print("=" * 50)
    
    # Test health first
    health_ok = test_health_endpoint()
    print()
    
    if health_ok:
        # Test CSV processing
        csv_ok = test_csv_endpoint()
        
        if csv_ok:
            print("\n🎉 All tests passed!")
            print("💾 Check for 'processed_listings.csv' file")
        else:
            print("\n⚠️  CSV processing test failed")
    else:
        print("\n⚠️  Health check failed - server may not be running")

if __name__ == "__main__":
    main()
