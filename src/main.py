import torch
import clip
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import time

# Load CLIP model once at startup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Loading CLIP model on {device.upper()}...")
model, preprocess = clip.load("ViT-L/14", device=device)
print("✅ Model loaded successfully!")

def detect_object_type(image_url):
    """Detect what type of object is in the image using CLIP image-only approach"""
    try:
        # Load image with proper headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(image_url, timeout=10, headers=headers)
        response.raise_for_status()
        
        # Try to open image
        image = Image.open(BytesIO(response.content))
        image = image.convert('RGB')  # Ensure RGB format
        
        # Preprocess image
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        # Extract image features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Use comprehensive object descriptions for detection
        object_descriptions = [
            "a scooter", "an electric scooter", "a kick scooter", "a razor scooter",
            "a bicycle", "a mountain bike", "a road bike", "an electric bike",
            "a motorcycle", "a moped", "a skateboard", "a hoverboard",
            "a car", "a truck", "a van", "a bus",
            "a chair", "a table", "a sofa", "a bed", "a couch",
            "a laptop", "a computer", "a phone", "a tablet",
            "a book", "a magazine", "a newspaper", "a document",
            "a toy", "a game", "a puzzle", "a doll",
            "a tool", "a hammer", "a screwdriver", "a wrench",
            "a kitchen appliance", "a refrigerator", "a microwave", "a toaster",
            "clothing", "a shirt", "a dress", "a jacket", "shoes", "sneakers",
            "a bag", "a backpack", "a purse", "a suitcase",
            "a camera", "a television", "a radio", "a speaker",
            "a plant", "a flower", "a tree", "a garden",
            "food", "a fruit", "a vegetable", "a meal",
            "a pet", "a dog", "a cat", "a bird",
            "a person", "a child", "a baby", "an adult"
        ]
        
        # Process with CLIP using text descriptions
        text = clip.tokenize(object_descriptions).to(device)

with torch.no_grad():
    text_features = model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            probs = similarity.cpu().numpy()
        
        # Get results
        results = list(zip(object_descriptions, probs[0]))
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return both object detection and image features
        return {
            'detected_object': results[0][0],  # Best object match
            'object_confidence': results[0][1],  # Confidence for best match
            'top_3_objects': results[:3],  # Top 3 object matches
            'image_features': image_features.cpu().numpy(),
            'feature_norm': torch.norm(image_features[0]).item(),
            'raw_features': image_features[0].cpu().numpy().tolist()
        }
        
    except Exception as e:
        print(f"❌ Error detecting object type: {e}")
        return {
            'detected_object': "unknown",
            'object_confidence': 0.0,
            'top_3_objects': [("unknown", 0.0)],
            'image_features': None,
            'feature_norm': 0.0,
            'raw_features': []
        }

def assess_condition(image_url, detected_object):
    """Assess the condition of the detected object"""
    try:
        # Load image with proper headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(image_url, timeout=10, headers=headers)
        response.raise_for_status()
        
        # Try to open image
        image = Image.open(BytesIO(response.content))
        image = image.convert('RGB')  # Ensure RGB format
        
        # Preprocess image
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        
        # Add object-specific conditions
        object_specific = [
            f"{detected_object} in like new condition",
            f"{detected_object} in excellent condition", 
            f"{detected_object} in good condition",
            f"{detected_object} in fair condition",
            f"{detected_object} in poor condition",
            f"{detected_object} needs repair",
            f"well maintained {detected_object}",
            f"damaged {detected_object}",
            f"broken {detected_object}"
        ]
        print(object_specific)
        
        # Combine all condition prompts
        all_conditions = object_specific
        
        # Process with CLIP
        text = clip.tokenize(all_conditions).to(device)

        with torch.no_grad():
            logits_per_image, logits_per_text = model(image_input, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        # Get results
        results = list(zip(all_conditions, probs[0]))
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results for more detailed analysis
        return {
            'top_match': results[0],
            'top_3': results[:3],
            'all_results': results
        }
        
    except Exception as e:
        print(f"❌ Error assessing condition: {e}")
        return {
            'top_match': ("unknown condition", 0.0),
            'top_3': [("unknown condition", 0.0)],
            'all_results': [("unknown condition", 0.0)]
        }

def process_csv_file(csv_path, max_items=None):
    """Process a CSV file with product data"""
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(df)} items from {csv_path}")
        
        if max_items:
            df = df.head(max_items)
            print(f"🔍 Processing first {max_items} items...")
        
        results = []
        
        for idx, row in df.iterrows():
            print(f"\n--- Processing Item {idx + 1}/{len(df)} ---")
            print(f"Title: {row['title']}")
            print(f"Price: {row['price']}")
            print(f"Location: {row['location_city']}, {row['location_state']}")
            
            # Step 1: Detect object type using comprehensive descriptions
            print("🔍 Detecting object type...")
            object_results = detect_object_type(row['photo_url'])
            detected_object = object_results['detected_object']
            object_confidence = object_results['object_confidence']
            top_3_objects = object_results['top_3_objects']
            print(f"   Detected: {detected_object} (confidence: {object_confidence:.3f})")
            print(f"   Top 3: {[f'{obj[0]} ({obj[1]:.3f})' for obj in top_3_objects]}")
            
            # Step 2: Assess condition using the detected object name
            print("🔍 Assessing condition...")
            condition_results = assess_condition(row['photo_url'], detected_object)
            top_condition = condition_results['top_match']
            top_3_conditions = condition_results['top_3']
            print(f"   Top: {top_condition[0]} (confidence: {top_condition[1]:.3f})")
            print(f"   Top 3: {[f'{cond[0]} ({cond[1]:.3f})' for cond in top_3_conditions]}")
            
            # Store results
            result = {
                'id': row['id'],
                'title': row['title'],
                'price': row['price'],
                'location': f"{row['location_city']}, {row['location_state']}",
                # Object detection results
                'detected_object': detected_object,
                'object_confidence': object_confidence,
                'detected_object_2nd': top_3_objects[1][0] if len(top_3_objects) > 1 else "N/A",
                'object_confidence_2nd': top_3_objects[1][1] if len(top_3_objects) > 1 else 0.0,
                'detected_object_3rd': top_3_objects[2][0] if len(top_3_objects) > 2 else "N/A",
                'object_confidence_3rd': top_3_objects[2][1] if len(top_3_objects) > 2 else 0.0,
                # Condition assessment results
                'condition': top_condition[0],
                'condition_confidence': top_condition[1],
                'condition_2nd': top_3_conditions[1][0] if len(top_3_conditions) > 1 else "N/A",
                'condition_confidence_2nd': top_3_conditions[1][1] if len(top_3_conditions) > 1 else 0.0,
                'condition_3rd': top_3_conditions[2][0] if len(top_3_conditions) > 2 else "N/A",
                'condition_confidence_3rd': top_3_conditions[2][1] if len(top_3_conditions) > 2 else 0.0,
                # Original data
                'url': row['url'],
                'photo_url': row['photo_url']
            }
            results.append(result)
            
            # Small delay to avoid overwhelming the server
            time.sleep(0.5)
        
        return results
        
    except Exception as e:
        print(f"❌ Error processing CSV: {e}")
        return []

def save_results(results, output_file="results.csv"):
    """Save results to CSV file"""
    if not results:
        print("❌ No results to save")
        return
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"💾 Results saved to {output_file}")
    
    # Print summary
    print(f"\n📈 SUMMARY:")
    print(f"   Total items processed: {len(results)}")
    
    # Object detection summary
    object_counts = df_results['detected_object'].value_counts()
    print(f"   Object types detected:")
    for obj, count in object_counts.head(5).items():
        print(f"     - {obj}: {count}")
    
    # Condition summary
    condition_counts = df_results['condition'].value_counts()
    print(f"   Top conditions:")
    for cond, count in condition_counts.head(5).items():
        print(f"     - {cond}: {count}")

def create_test_csv():
    """Create a test CSV with random images for testing"""
    print("📝 Creating test CSV with random images...")
    
    # Random images from various sources (public domain/creative commons)
    test_data = [
        {
            "title": "Vintage Bicycle",
            "price": "$150",
            "location_city": "Portland",
            "location_state": "OR",
            "id": "test_001",
            "url": "https://example.com/test_001",
            "photo_url": "https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=400",
            "miles": "N/A"
        },
        {
            "title": "Couch",
            "price": "$150",
            "location_city": "Portland",
            "location_state": "OR",
            "id": "test_002",
            "url": "https://example.com/test_002",
            "photo_url": "https://scontent-lax3-1.xx.fbcdn.net/v/t45.5328-4/540346486_798106729320942_231890133157320701_n.jpg?stp=c0.151.261.261a_dst-jpg_p261x260_tt6&_nc_cat=102&ccb=1-7&_nc_sid=247b10&_nc_ohc=1LSHKss4cFsQ7kNvwEJgCDB&_nc_oc=AdnABV6EGKsCki2uMvG_7aGhQM0Zx7FSgthgdSEIbAC6vYfsaxcRUO36L6gPkNMr1zN5oxCUDqbgFPmXfkXiDewM&_nc_zt=23&_nc_ht=scontent-lax3-1.xx&_nc_gid=vIN0efnMmHa-yJGN__XVJQ&oh=00_AfV_pME-0BoCBQkDRx5qe8S3deUj34hUZVFQf2OaBk43AQ&oe=68BC6B80",
            "miles": "N/A"
        }, {
            "title": "poor couch",
            "price": "$150",  
            "location_city": "New England",
            "location_state": "OR",
            "id": "test_003",
            "url": "https://example.com/test_003",
            "photo_url": "https://scontent-lax3-1.xx.fbcdn.net/v/t45.5328-4/541549792_4215891432070535_4657146202498903696_n.jpg?stp=c151.0.260.260a_dst-jpg_p261x260_tt6&_nc_cat=104&ccb=1-7&_nc_sid=247b10&_nc_ohc=CNnkRGfYVtsQ7kNvwGmv3D6&_nc_oc=AdluJPkBCMO7JlKz6CDt-bQmrKeYdp1QKiMm1V9diBfQ9luKyk0PuloHKzC_IjMdVrB6sesrhDWtzbGHc0ozKOXJ&_nc_zt=23&_nc_ht=scontent-lax3-1.xx&_nc_gid=_NKLz-rEFJBrmAXyF-1zLA&oh=00_AfX9U3zSahpbQcqo_voodUEV-k_ELGXMhA3sNMjoaSGSUA&oe=68BC7BBD",
            "miles": "N/A"
        }, {
            "title": "third supreme",
            "price": "$150",  
            "location_city": "New England",
            "location_state": "OR",
            "id": "test_003",
            "url": "https://example.com/test_003",
            "photo_url": "https://scontent-lax3-1.xx.fbcdn.net/v/t45.5328-4/541428204_785240323869229_5798388497150660454_n.jpg?stp=c0.43.261.261a_dst-jpg_p261x260_tt6&_nc_cat=110&ccb=1-7&_nc_sid=247b10&_nc_ohc=yLHGyhw3RA8Q7kNvwFjSTb3&_nc_oc=AdlFeAVBTrIoopT2b400NO7DsY2B_hV4XfzF1RFwHVs7JRCgdE3W34jnk8vbhHJnTvhgxjObjE1sfj9NOLFDQM3U&_nc_zt=23&_nc_ht=scontent-lax3-1.xx&_nc_gid=tVnVp7HKj5NUlURMDWVBmA&oh=00_AfWoL3X8JAhtbSQ9Bz-Me3yXSG3Fr7Fv18EmpKqASZa2lA&oe=68BC79EA",
            "miles": "N/A"
        },
        {
            "title": "fourth image",
            "price": "$150",  
            "location_city": "New England",
            "location_state": "OR",
            "id": "test_003",
            "url": "https://example.com/test_003",
            "photo_url": "https://scontent-lax3-2.xx.fbcdn.net/v/t45.5328-4/528623575_761914299717478_5763832977874385031_n.jpg?stp=c0.0.261.261a_dst-jpg_p261x260_tt6&_nc_cat=111&ccb=1-7&_nc_sid=247b10&_nc_ohc=qEHSlyzUiKEQ7kNvwG_tcwf&_nc_oc=Adkh4jFFGSgzP9JLXXKH41I6s_0rMt2cM5fT6AorHC60TaX83-Y-_7vtcV4EyAYqNoN18IwSCVdsTUzvP9alK76e&_nc_zt=23&_nc_ht=scontent-lax3-2.xx&_nc_gid=_NKLz-rEFJBrmAXyF-1zLA&oh=00_AfX2vMD0xbUSBNr-RdaY8MRsJUeaV9xlfzVEMV5Jphw_1A&oe=68BC71C5",
            "miles": "N/A"
        },
        {
            "title": "fifth nike",
            "price": "$150",  
            "location_city": "New England",
            "location_state": "OR",
            "id": "test_003",
            "url": "https://example.com/test_003",
            "photo_url": "https://scontent-lax3-1.xx.fbcdn.net/v/t45.5328-4/540310722_779810248549698_6426088553411326993_n.jpg?stp=c151.0.260.260a_dst-jpg_p261x260_tt6&_nc_cat=110&ccb=1-7&_nc_sid=247b10&_nc_ohc=qiMl568DRzoQ7kNvwGSekjU&_nc_oc=AdnRr75M8arVCwmWBQISNp79JftPKF53-gKDWsxhETu92QycMhVfKGoG_FMHSXb61-myB-yhbz0ZDZgRVqOGivAo&_nc_zt=23&_nc_ht=scontent-lax3-1.xx&_nc_gid=0-EKM_6iyNuv7Jalq4liqQ&oh=00_AfWWxKnNIyX62Dp7WpboQxJE2PrxX159NI3IqbwM-ipAGw&oe=68BC7E76",
            "miles": "N/A"
        },
        {
            "title": "sixth addidas",
            "price": "$150",  
            "location_city": "New England",
            "location_state": "OR",
            "id": "test_003",
            "url": "https://example.com/test_003",
            "photo_url": "https://scontent-lax3-1.xx.fbcdn.net/v/t45.5328-4/532940724_1297016961801725_1815344610964279089_n.jpg?stp=c0.43.261.261a_dst-jpg_p261x260_tt6&_nc_cat=110&ccb=1-7&_nc_sid=247b10&_nc_ohc=hZHadHqP9zsQ7kNvwFB8YdW&_nc_oc=AdnZGO2PESRFVQX0-nH3N0mIaa9g6GDQ-jnKJ8c3icbkvb8bvisfHoMfyJcz_JsnUBMxVXqbKRmmB3aeqY6W-SAg&_nc_zt=23&_nc_ht=scontent-lax3-1.xx&_nc_gid=0-EKM_6iyNuv7Jalq4liqQ&oh=00_AfX6ldUyexA14KocM_o8llSpljNx6lEDBGZfGbPm4Hztkg&oe=68BC6A7E",
            "miles": "N/A"
        }
    ]
    # Create DataFrame and save
    df = pd.DataFrame(test_data)
    df.to_csv("test_items.csv", index=False)
    print("✅ Test CSV created: test_items.csv")
    return "test_items.csv"

def main():
    """Main function to run the pipeline"""
    print("🎯 GoodDeals Condition Detection Pipeline")
    print("=" * 50)
    
    
    print("\n" + "=" * 50)
    print("📊 Creating test data and processing...")
    
    # Create test CSV with random images
    csv_file = create_test_csv()
    
    # Process the test CSV
    results = process_csv_file(csv_file, max_items=None)  # Process all items
    
    if results:
        save_results(results, "final_results.csv")
        
        # Show detailed results in table format
        print(f"\n📋 DETAILED RESULTS:")
        print(f"{'Item':<20} {'Detected Object':<20} {'Confidence':<12} {'Condition':<30} {'Confidence':<12}")
        print("-" * 100)
        for i, result in enumerate(results, 1):
            print(f"{result['title']:<20} {result['detected_object']:<20} {result['object_confidence']:<12.3f} {result['condition']:<30} {result['condition_confidence']:<12.3f}")
        
        print(f"\n📊 ADDITIONAL DETAILS:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']}")
            print(f"   Object 2nd: {result['detected_object_2nd']} ({result['object_confidence_2nd']:.3f})")
            print(f"   Object 3rd: {result['detected_object_3rd']} ({result['object_confidence_3rd']:.3f})")
            print(f"   Condition 2nd: {result['condition_2nd']} ({result['condition_confidence_2nd']:.3f})")
            print(f"   Condition 3rd: {result['condition_3rd']} ({result['condition_confidence_3rd']:.3f})")
            print(f"   Price: {result['price']}")
    else:
        print("❌ No results generated")

if __name__ == "__main__":
    main()