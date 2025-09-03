from io import BytesIO
import time

from PIL import Image
import clip
from gpt4all import GPT4All
import pandas as pd
import requests
import torch

# Load CLIP model once at startup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Loading CLIP model on {device.upper()}...")
model, preprocess = clip.load("ViT-L/14", device=device)
print("✅ CLIP model loaded successfully!")

# Load Llama model for object detection
print(f"🤖 Loading Llama 3 8B model...")
llama_model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")
print("✅ Llama model loaded successfully!")

def detect_object_with_llama(title):
    """Use local Llama 3 8B model to detect object type from title"""
    try:
        # Use GPT4All with Llama model
        with llama_model.chat_session():
            prompt = f"""You are a tagger. From the given listing title, output EXACTLY ONE generic, neutral, singular NOUN that best describes the item. 
Rules:
- Output must be ONE WORD only, lowercase letters only (^[a-z]+$). 
- No sentences, no quotes, no adjectives, no sentiment (no "trash/garbage/junk"), no verbs, no punctuation, no filler.
- Ignore condition, urgency, price, free/giveaway phrasing, and calls to action (e.g., "used", "need gone", "free", "asap", "pick up").
- If the title is ambiguous, output item.
- Prefer the umbrella hypernym if multiple items are referenced (e.g., chairs and table → furniture).
- Canonicalize common synonyms (cellphone → phone, tv → television, laptop → laptop).
Respond with the single word only to describe: {title}"""
            response = llama_model.generate(prompt, max_tokens=10)
        
        # Clean up the response
        object_type = response.strip().lower()
        object_type = object_type.replace('.', '').replace(',', '').replace('\n', '').strip()
        
        # Extract just the first word if multiple words returned
        object_type = object_type.split()[0] if object_type.split() else "item"
        
        return {
            'detected_object': f"a {object_type}",
            'object_confidence': 0.0,  # Llama doesn't provide confidence scores (unlike CLIP)
            'detection_method': 'llama3_8b',
            'raw_response': response
        }
            
    except Exception as e:
        print(f"❌ Error with Llama detection: {e}")
        print("🔄 Falling back to simple keyword matching...")
        # Fallback to simple keyword matching
        return detect_object_from_title_simple(title)

def detect_object_from_title_simple(title):
    """Simple fallback object detection using keyword matching"""
    try:
        title_lower = title.lower()
        
        # Simple keyword mapping
        keyword_mapping = {
            'bicycle': ['bike', 'bicycle', 'road bike', 'mountain bike', 'bmx', 'cruiser', 'hybrid', 'trek', 'specialized', 'giant', 'cannondale'],
            'sofa': ['sofa', 'couch', 'sectional', 'loveseat', 'futon'],
            'chair': ['chair', 'dining chair', 'office chair', 'recliner', 'armchair'],
            'table': ['table', 'dining table', 'coffee table', 'end table', 'desk'],
            'bed': ['bed', 'mattress', 'bedframe', 'headboard'],
            'laptop': ['laptop', 'macbook', 'dell', 'hp', 'lenovo', 'asus'],
            'phone': ['phone', 'iphone', 'samsung', 'android', 'smartphone', 'galaxy'],
            'tablet': ['tablet', 'ipad', 'kindle', 'surface'],
            'tv': ['tv', 'television', 'monitor', 'screen', 'display'],
            'shoes': ['shoes', 'sneakers', 'boots', 'sandals', 'nike', 'adidas', 'jordan'],
            'car': ['car', 'vehicle', 'sedan', 'suv', 'truck', 'van', 'bmw', 'mercedes', 'audi', 'honda', 'toyota'],
            'tool': ['tool', 'hammer', 'screwdriver', 'wrench', 'drill', 'saw'],
            'appliance': ['refrigerator', 'microwave', 'oven', 'dishwasher', 'washer', 'dryer'],
        }
        
        # Check for matches
        for category, keywords in keyword_mapping.items():
            if any(keyword in title_lower for keyword in keywords):
                return {
                    'detected_object': f"a {category}",
                    'object_confidence': 0.7,  # Medium confidence for keyword matching
                    'detection_method': 'keyword_fallback',
                    'matched_keywords': [kw for kw in keywords if kw in title_lower]
                }
        
        # Ultimate fallback
        return {
            'detected_object': "an item",
            'object_confidence': 0.1,
            'detection_method': 'unknown',
            'original_title': title
        }
        
    except Exception as e:
        print(f"❌ Error in fallback detection: {e}")
        return {
            'detected_object': "unknown",
            'object_confidence': 0.0,
            'detection_method': 'error'
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
        
        
        # # Add object-specific conditions
        # object_specific = [
        #     f"{detected_object} in like new condition",
        #     f"{detected_object} in excellent condition", 
        #     f"{detected_object} in good condition",
        #     f"{detected_object} in fair condition",
        #     f"{detected_object} in poor condition",
        #     f"{detected_object} needs repair",
        #     f"well maintained {detected_object}",
        #     f"damaged {detected_object}",
        #     f"broken {detected_object}"
        # ]
        object_specific = [
            f"{detected_object} in good condition",
            f"{detected_object} in fair condition",
            f"{detected_object} in poor condition"
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
            
            # Step 1: Detect object type from title using Llama 3 8B AI
            print("🔍 Detecting object type from title using Llama 3 8B...")
            object_results = detect_object_with_llama(row['title'])
            detected_object = object_results['detected_object']
            object_confidence = object_results['object_confidence']
            detection_method = object_results['detection_method']
            print(f"   Detected: {detected_object} (confidence: {object_confidence:.3f})")
            print(f"   Method: {detection_method}")
            if 'matched_brand' in object_results:
                print(f"   Brand: {object_results['matched_brand']}")
            if 'matched_keywords' in object_results:
                print(f"   Keywords: {object_results['matched_keywords']}")
            
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
                # Object detection results (from title)
                'detected_object': detected_object,
                'object_confidence': object_confidence,
                'detection_method': detection_method,
                'raw_response': object_results.get('raw_response', 'N/A'),
                'matched_keywords': ', '.join(object_results.get('matched_keywords', [])),
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


def main():
    """Main function to run the pipeline"""
    print("🎯 GoodDeals Condition Detection Pipeline")
    print("=" * 50)
    
    
    print("\n" + "=" * 50)
    print("📊 Creating test data and processing...")
    
    # Process the test CSV
    results = process_csv_file("test_list.csv", max_items=None)  # Process all items from test_items.csv
    
    if results:
        save_results(results, "final_results.csv")
        
        # Show detailed results in table format
        print(f"\n📋 DETAILED RESULTS:")
        print(f"{'Item':<20} {'Detected Object':<20} {'Confidence':<12} {'Condition':<30} {'Confidence':<12}")
        print("-" * 100)
        for i, result in enumerate(results, 1):
            print(f"{result['title']:<20} {result['detected_object']:<20} {result['object_confidence']:<12.3f} {result['condition']:<30} {result['condition_confidence']:<12.3f}")
        
        print(f"\n📊 ADDITIONAL DETAILS:")
        for i, result in enumerate(results, 0):
            print(f"\n{i}. {result['title']}")
            print(f"   Detection Method: {result['detection_method']}")
            if result['raw_response'] != 'N/A':
                print(f"   AI Response: {result['raw_response']}")
            if result['matched_keywords']:
                print(f"   Matched Keywords: {result['matched_keywords']}")
            print(f"   Condition 1st: {result['condition']} ({result['condition_confidence']:.3f})")
            print(f"   Condition 2nd: {result['condition_2nd']} ({result['condition_confidence_2nd']:.3f})")
            print(f"   Condition 3rd: {result['condition_3rd']} ({result['condition_confidence_3rd']:.3f})")
            print(f"   Price: {result['price']}")
    else:
        print("❌ No results generated")

if __name__ == "__main__":
    main()