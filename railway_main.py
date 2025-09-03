from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio
import aiohttp
from io import BytesIO
import time

# Initialize FastAPI
app = FastAPI(title="GoodDeals Condition API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for lazy loading
model = None
preprocess = None
llama_model = None
device = None

async def load_models():
    """Load ML models on first request"""
    global model, preprocess, llama_model, device
    
    if model is None:
        try:
            import torch
            import clip
            from gpt4all import GPT4All
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading CLIP model on {device}...")
            model, preprocess = clip.load("ViT-L/14", device=device)
            
            print("Loading Llama model...")
            llama_model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")
            
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise HTTPException(status_code=500, detail="Failed to load ML models")

# Pydantic models for request/response
class ListingItem(BaseModel):
    id: str
    title: str
    price: str
    source: str
    image: str
    location: str
    url: str
    miles: str

class ConditionRequest(BaseModel):
    results: List[ListingItem]

class EnhancedListingItem(ListingItem):
    detected_object: str
    object_confidence: float
    condition: str
    condition_confidence: float
    detection_method: str

class ConditionResponse(BaseModel):
    results: List[EnhancedListingItem]
    processing_time: float
    total_processed: int

async def detect_object_with_llama(title: str) -> Dict[str, Any]:
    """Async object detection with Llama"""
    try:
        loop = asyncio.get_event_loop()
        
        def run_llama():
            with llama_model.chat_session():
                prompt = f"""You are a tagger. From the given listing title, output EXACTLY ONE generic, neutral, singular NOUN that best describes the item. 
Rules:
- Output must be ONE WORD only, lowercase letters only (^[a-z]+$). 
- No sentences, no quotes, no adjectives, no sentiment, no verbs, no punctuation.
- Ignore condition, urgency, price, free/giveaway phrasing.
- If ambiguous, output item.
Respond with the single word only to describe: {title}"""
                return llama_model.generate(prompt, max_tokens=10)
        
        # Run in thread pool to avoid blocking
        response = await loop.run_in_executor(None, run_llama)
        
        object_type = response.strip().lower()
        object_type = object_type.replace('.', '').replace(',', '').replace('\n', '').strip()
        object_type = object_type.split()[0] if object_type.split() else "item"
        
        return {
            'detected_object': f"a {object_type}",
            'object_confidence': 0.8,
            'detection_method': 'llama3_8b',
        }
    except Exception as e:
        print(f"Llama error: {e}")
        return detect_object_fallback(title)

def detect_object_fallback(title: str) -> Dict[str, Any]:
    """Fast fallback detection"""
    title_lower = title.lower()
    
    keyword_mapping = {
        'bicycle': ['bike', 'bicycle', 'trek', 'specialized'],
        'sofa': ['sofa', 'couch', 'sectional', 'loveseat'],
        'chair': ['chair', 'recliner', 'armchair'],
        'table': ['table', 'desk'],
        'laptop': ['laptop', 'macbook', 'dell', 'hp'],
        'phone': ['phone', 'iphone', 'samsung', 'android'],
        'tv': ['tv', 'television', 'monitor'],
        'car': ['car', 'vehicle', 'sedan', 'suv', 'bmw', 'honda'],
    }
    
    for category, keywords in keyword_mapping.items():
        if any(keyword in title_lower for keyword in keywords):
            return {
                'detected_object': f"a {category}",
                'object_confidence': 0.7,
                'detection_method': 'keyword_fallback',
            }
    
    return {
        'detected_object': "an item",
        'object_confidence': 0.5,
        'detection_method': 'unknown',
    }

async def assess_condition_async(image_url: str, detected_object: str) -> Dict[str, Any]:
    """Async condition assessment"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Async image download
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url, headers=headers, timeout=10) as response:
                if response.status == 200:
                    image_data = await response.read()
                else:
                    raise Exception(f"HTTP {response.status}")
        
        # Process image
        loop = asyncio.get_event_loop()
        
        def process_image():
            image = Image.open(BytesIO(image_data)).convert('RGB')
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
            
            return results[0]  # Return top match
        
        top_match = await loop.run_in_executor(None, process_image)
        
        return {
            'condition': top_match[0],
            'condition_confidence': float(top_match[1])
        }
        
    except Exception as e:
        print(f"Condition assessment error: {e}")
        return {
            'condition': f"{detected_object} in unknown condition",
            'condition_confidence': 0.0
        }

@app.post("/assess-conditions", response_model=ConditionResponse)
async def assess_conditions(request: ConditionRequest):
    """Main endpoint to assess conditions for listings"""
    start_time = time.time()
    
    # Load models on first request
    await load_models()
    
    try:
        enhanced_results = []
        
        # Process items concurrently (but limit concurrency to avoid overwhelming)
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent processes
        
        async def process_item(item: ListingItem):
            async with semaphore:
                # Step 1: Detect object
                object_data = await detect_object_with_llama(item.title)
                
                # Step 2: Assess condition  
                condition_data = await assess_condition_async(item.image, object_data['detected_object'])
                
                # Combine data
                enhanced_item = EnhancedListingItem(
                    **item.dict(),
                    detected_object=object_data['detected_object'],
                    object_confidence=object_data['object_confidence'],
                    condition=condition_data['condition'],
                    condition_confidence=condition_data['condition_confidence'],
                    detection_method=object_data['detection_method']
                )
                
                return enhanced_item
        
        # Process all items concurrently
        tasks = [process_item(item) for item in request.results]
        enhanced_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out any exceptions
        valid_results = [r for r in enhanced_results if not isinstance(r, Exception)]
        
        processing_time = time.time() - start_time
        
        return ConditionResponse(
            results=valid_results,
            processing_time=processing_time,
            total_processed=len(valid_results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": device,
        "models_loaded": model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)