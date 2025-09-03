#!/usr/bin/env python3
"""
Test script for ConditionAI API
"""

import requests
import json
import time

# Configuration
API_URL = "http://localhost:8000"  # Change this to your Oracle Cloud IP
# API_URL = "http://your-oracle-ip:8000"

def test_health():
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")

def test_direct_processing():
    """Test direct processing endpoint"""
    print("\n🔍 Testing direct processing...")
    
    # Sample test data
    test_items = [
        {
            "id": "test_001",
            "title": "Couch for sale for free",
            "price": "$150",
            "location_city": "Portland",
            "location_state": "OR",
            "url": "https://example.com/test_001",
            "photo_url": "https://scontent-lax3-1.xx.fbcdn.net/v/t45.5328-4/540346486_798106729320942_231890133157320701_n.jpg?stp=c0.151.261.261a_dst-jpg_p261x260_tt6&_nc_cat=102&ccb=1-7&_nc_sid=247b10&_nc_ohc=1LSHKss4cFsQ7kNvwEJgCDB&_nc_oc=AdnABV6EGKsCki2uMvG_7aGhQM0Zx7FSgthgdSEIbAC6vYfsaxcRUO36L6gPkNMr1zN5oxCUDqbgFPmXfkXiDewM&_nc_zt=23&_nc_ht=scontent-lax3-1.xx&_nc_gid=vIN0efnMmHa-yJGN__XVJQ&oh=00_AfV_pME-0BoCBQkDRx5qe8S3deUj34hUZVFQf2OaBk43AQ&oe=68BC6B80",
            "miles": "N/A"
        }
    ]
    
    try:
        response = requests.post(
            f"{API_URL}/process-direct",
            json={"items": test_items, "max_items": 1},
            timeout=120  # 2 minutes timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Direct processing successful")
            print(f"   Processed {len(data['results'])} items")
            
            if data['results']:
                result = data['results'][0]
                print(f"   Item: {result['title']}")
                print(f"   Detected: {result['detected_object']}")
                print(f"   Condition: {result['condition']}")
                print(f"   Confidence: {result['condition_confidence']:.3f}")
        else:
            print(f"❌ Direct processing failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Direct processing error: {e}")

def test_async_processing():
    """Test async processing endpoint"""
    print("\n🔍 Testing async processing...")
    
    test_items = [
        {
            "id": "test_002",
            "title": "Used couch for sale asap",
            "price": "$150",
            "location_city": "New England",
            "location_state": "OR",
            "url": "https://example.com/test_002",
            "photo_url": "https://scontent-lax3-1.xx.fbcdn.net/v/t45.5328-4/541549792_4215891432070535_4657146202498903696_n.jpg?stp=c151.0.260.260a_dst-jpg_p261x260_tt6&_nc_cat=104&ccb=1-7&_nc_sid=247b10&_nc_ohc=CNnkRGfYVtsQ7kNvwGmv3D6&_nc_oc=AdluJPkBCMO7JlKz6CDt-bQmrKeYdp1QKiMm1V9diBfQ9luKyk0PuloHKzC_IjMdVrB6sesrhDWtzbGHc0ozKOXJ&_nc_zt=23&_nc_ht=scontent-lax3-1.xx&_nc_gid=_NKLz-rEFJBrmAXyF-1zLA&oh=00_AfX9U3zSahpbQcqo_voodUEV-k_ELGXMhA3sNMjoaSGSUA&oe=68BC7BBD",
            "miles": "N/A"
        }
    ]
    
    try:
        # Start async job
        response = requests.post(
            f"{API_URL}/process",
            json={"items": test_items, "max_items": 1}
        )
        
        if response.status_code == 200:
            data = response.json()
            job_id = data['job_id']
            print(f"✅ Async job started: {job_id}")
            
            # Poll for completion
            max_attempts = 30  # 5 minutes max
            for attempt in range(max_attempts):
                time.sleep(10)  # Wait 10 seconds
                
                status_response = requests.get(f"{API_URL}/job/{job_id}")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"   Status: {status_data['status']} ({status_data['progress']}%)")
                    
                    if status_data['status'] == 'completed':
                        print("✅ Async processing completed")
                        if status_data['results']:
                            result = status_data['results'][0]
                            print(f"   Item: {result['title']}")
                            print(f"   Detected: {result['detected_object']}")
                            print(f"   Condition: {result['condition']}")
                            print(f"   Confidence: {result['condition_confidence']:.3f}")
                        break
                    elif status_data['status'] == 'failed':
                        print(f"❌ Async processing failed: {status_data.get('error_message', 'Unknown error')}")
                        break
                else:
                    print(f"❌ Failed to get job status: {status_response.status_code}")
                    break
            else:
                print("⏰ Async processing timed out")
                
        else:
            print(f"❌ Failed to start async job: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Async processing error: {e}")

def main():
    """Run all tests"""
    print("🧪 ConditionAI API Test Suite")
    print("=" * 50)
    
    test_health()
    test_direct_processing()
    test_async_processing()
    
    print("\n" + "=" * 50)
    print("🏁 Test suite completed")

if __name__ == "__main__":
    main()
