#!/usr/bin/env python3
"""
Test the fixed railway_main.py locally
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from railway_main import GoodDealsConditionAPI, ListingItem
import pandas as pd

def test_api():
    """Test the API with a simple item"""
    print("🧪 Testing fixed API...")
    
    # Create API instance
    api = GoodDealsConditionAPI()
    
    # Initialize with CPU device
    api.setup("cpu")
    
    # Test with a simple item
    test_item = ListingItem(
        id="test_001",
        title="Bike for sale",
        price="$50",
        source="test",
        image="https://scontent-lax3-1.xx.fbcdn.net/v/t45.5328-4/541671374_1430551655093177_7488556763206710628_n.jpg",
        location="San Diego, CA",
        url="https://example.com/test"
    )
    
    print(f"📥 Testing with item: {test_item.title}")
    
    # Test processing
    try:
        result = api._process_item_sync(test_item)
        print("✅ Processing successful!")
        print(f"   Detected object: {result.detected_object}")
        print(f"   Object confidence: {result.object_confidence}")
        print(f"   Detection method: {result.detection_method}")
        print(f"   Condition: {result.condition}")
        print(f"   Condition confidence: {result.condition_confidence}")
        return True
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_csv_processing():
    """Test CSV processing"""
    print("\n🧪 Testing CSV processing...")
    
    try:
        # Read CSV
        df = pd.read_csv("listings_from_txt.csv")
        print(f"📊 Loaded {len(df)} items from CSV")
        
        # Test with first item
        first_row = df.iloc[0]
        test_item = ListingItem(
            id=str(first_row['id']),
            title=str(first_row['title']),
            price=str(first_row['price']),
            source="csv",
            image=str(first_row['photo_url']) if pd.notna(first_row['photo_url']) else None,
            location=f"{first_row['location_city']}, {first_row['location_state']}",
            url=str(first_row['url'])
        )
        
        print(f"📥 Testing with CSV item: {test_item.title}")
        
        # Create API and test
        api = GoodDealsConditionAPI()
        api.setup("cpu")
        
        result = api._process_item_sync(test_item)
        print("✅ CSV processing successful!")
        print(f"   Detected object: {result.detected_object}")
        print(f"   Object confidence: {result.object_confidence}")
        print(f"   Detection method: {result.detection_method}")
        return True
        
    except Exception as e:
        print(f"❌ CSV processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing Fixed Railway Main")
    print("=" * 50)
    
    # Test basic API
    api_ok = test_api()
    
    # Test CSV processing
    csv_ok = test_csv_processing()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    print(f"  API Test: {'✅ PASS' if api_ok else '❌ FAIL'}")
    print(f"  CSV Test: {'✅ PASS' if csv_ok else '❌ FAIL'}")
    
    if api_ok and csv_ok:
        print("\n🎉 All tests passed! The fix works.")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
