#!/usr/bin/env python3
"""
Direct CSV processing script - no API, no async complexity
Processes listings_from_txt.csv directly
"""

import pandas as pd
import requests
import time
from io import BytesIO
from PIL import Image
import torch
import clip
import json

# Load models at startup
print("🚀 Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
print("✅ CLIP model loaded successfully!")

def detect_object_fallback(title: str) -> dict:
    """Enhanced fallback detection with more keywords"""
    title_lower = title.lower()
    
    keyword_mapping = {
        'bicycle': ['bike', 'bicycle', 'cycling', 'trek', 'specialized', 'cannondale', 'giant', 'road bike', 'mountain bike', 'cruiser', 'huffy', 'schwinn', 'kulana'],
        'sofa': ['sofa', 'couch', 'sectional', 'loveseat', 'settee'],
        'chair': ['chair', 'recliner', 'armchair', 'office chair', 'gaming chair', 'dining chair'],
        'table': ['table', 'desk', 'dining table', 'coffee table', 'end table', 'nightstand'],
        'laptop': ['laptop', 'macbook', 'dell', 'hp', 'lenovo', 'computer', 'notebook'],
        'phone': ['phone', 'iphone', 'samsung', 'android', 'mobile', 'smartphone', 'cell phone'],
        'tv': ['tv', 'television', 'monitor', 'screen', 'display', 'smart tv'],
        'car': ['car', 'vehicle', 'sedan', 'suv', 'bmw', 'honda', 'toyota', 'truck', 'auto'],
        'watch': ['watch', 'timepiece', 'rolex', 'apple watch', 'smartwatch', 'wristwatch'],
        'guitar': ['guitar', 'acoustic', 'electric', 'bass', 'fender', 'gibson'],
        'camera': ['camera', 'nikon', 'canon', 'sony', 'photography', 'dslr', 'lens'],
        'book': ['book', 'novel', 'textbook', 'manual', 'hardcover', 'paperback'],
        'shoes': ['shoes', 'sneakers', 'boots', 'sandals', 'nike', 'adidas', 'footwear'],
        'jacket': ['jacket', 'coat', 'hoodie', 'blazer', 'sweater', 'cardigan'],
        'dresser': ['dresser', 'drawer', 'chest', 'bureau'],
        'bed': ['bed', 'mattress', 'bedframe', 'queen', 'king', 'twin'],
    }
    
    # Check for exact matches first
    for category, keywords in keyword_mapping.items():
        if any(keyword in title_lower for keyword in keywords):
            return {
                'detected_object': f"a {category}",
                'object_confidence': 0.82,
                'detection_method': 'enhanced_keyword_fallback',
            }
    
    # If no match, try to extract likely nouns
    import re
    words = re.findall(r'\b[a-zA-Z]+\b', title_lower)
    skip_words = ['free', 'sale', 'good', 'great', 'nice', 'condition', 'used', 'new', 'excellent', 'fair', 'poor', 'adult', 'boys', 'girls']
    
    for word in words:
        if len(word) > 3 and word not in skip_words and word.isalpha():
            return {
                'detected_object': f"a {word}",
                'object_confidence': 0.65,
                'detection_method': 'noun_extraction',
            }
    
    return {
        'detected_object': "an item",
        'object_confidence': 0.5,
        'detection_method': 'unknown',
    }

def assess_condition(image_url: str, detected_object: str) -> dict:
    """Assess condition using CLIP"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Download image
        response = requests.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Process image
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        conditions = [
            f"specifically for {detected_object} in excellent condition",
            f"specifically for {detected_object} in good condition", 
            f"specifically for {detected_object} in fair condition",
            f"specifically for {detected_object} in poor condition"
        ]
        
        text = clip.tokenize(conditions).to(device)
        
        with torch.no_grad():
            logits_per_image, logits_per_text = model(image_input, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
        results = list(zip(conditions, probs[0]))
        results.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'condition': results[0][0],
            'condition_confidence': float(results[0][1])
        }
        
    except Exception as e:
        print(f"❌ Condition assessment error for {image_url}: {e}")
        return {
            'condition': f"{detected_object} in unknown condition",
            'condition_confidence': 0.0
        }

def process_csv(csv_file: str, max_items: int = None):
    """Process CSV file directly"""
    print(f"📊 Loading CSV: {csv_file}")
    df = pd.read_csv(csv_file)
    
    if max_items:
        df = df.head(max_items)
        print(f"🔍 Processing first {max_items} items...")
    else:
        print(f"🔍 Processing all {len(df)} items...")
    
    results = []
    
    for idx, row in df.iterrows():
        print(f"\n--- Processing Item {idx + 1}/{len(df)} ---")
        print(f"Title: {row['title']}")
        print(f"Price: {row['price']}")
        print(f"Location: {row['location_city']}, {row['location_state']}")
        
        # Step 1: Detect object type
        print("🔍 Detecting object type...")
        object_data = detect_object_fallback(row['title'])
        print(f"   Detected: {object_data['detected_object']} (confidence: {object_data['object_confidence']:.3f})")
        
        # Step 2: Assess condition
        print("🔍 Assessing condition...")
        if pd.notna(row['photo_url']) and row['photo_url']:
            condition_data = assess_condition(row['photo_url'], object_data['detected_object'])
        else:
            condition_data = {
                'condition': f"{object_data['detected_object']} in unknown condition",
                'condition_confidence': 0.0
            }
        print(f"   Condition: {condition_data['condition']} (confidence: {condition_data['condition_confidence']:.3f})")
        
        # Store results
        result = {
            'id': row['id'],
            'title': row['title'],
            'price': row['price'],
            'location': f"{row['location_city']}, {row['location_state']}",
            'detected_object': object_data['detected_object'],
            'object_confidence': object_data['object_confidence'],
            'detection_method': object_data['detection_method'],
            'condition': condition_data['condition'],
            'condition_confidence': condition_data['condition_confidence'],
            'url': row['url'],
            'photo_url': row['photo_url']
        }
        results.append(result)
        
        # Small delay to avoid overwhelming
        time.sleep(0.5)
    
    return results

def save_results(results, output_file="processed_results.csv"):
    """Save results to CSV"""
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
    for obj, count in object_counts.head(10).items():
        print(f"     - {obj}: {count}")
    
    # Condition summary
    condition_counts = df_results['condition'].value_counts()
    print(f"   Top conditions:")
    for cond, count in condition_counts.head(10).items():
        print(f"     - {cond}: {count}")

def main():
    """Main function"""
    print("🎯 Direct CSV Processing - No API Complexity")
    print("=" * 60)
    
    # Process the CSV
    results = process_csv("listings_from_txt.csv", max_items=10)  # Process first 10 items
    
    if results:
        save_results(results, "processed_listings.csv")
        
        # Show detailed results
        print(f"\n📋 DETAILED RESULTS:")
        print(f"{'Item':<30} {'Detected Object':<20} {'Confidence':<12} {'Condition':<40} {'Confidence':<12}")
        print("-" * 120)
        for result in results:
            print(f"{result['title'][:29]:<30} {result['detected_object']:<20} {result['object_confidence']:<12.3f} {result['condition'][:39]:<40} {result['condition_confidence']:<12.3f}")
    else:
        print("❌ No results generated")

if __name__ == "__main__":
    main()
