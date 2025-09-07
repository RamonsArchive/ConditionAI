"""
Condition Assessment API for GoodDeals
- Uses LitServe for efficient serving
- Preloads CLIP model for fast inference
- Integrates with Ollama for object detection
- Automatically saves results to CSV
"""

from litserve import LitAPI, LitServer
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import time
from PIL import Image
import requests
from io import BytesIO
import torch
import clip

class ListingItem(BaseModel):
    id: str
    title: str
    price: Optional[Union[str, float, int]] = None
    source: Optional[str] = None
    image: Optional[str] = None
    location: Optional[str] = None
    url: Union[str, Any, None] = None
    miles: Optional[Union[str, float, int]] = None

class ConditionRequest(BaseModel):
    search_query: str
    results: List[ListingItem]

class EnhancedListingItem(ListingItem):
    detected_object: str
    condition: str
    condition_confidence: float

class ConditionResponse(BaseModel):
    results: List[EnhancedListingItem]
    processing_time: float
    total_processed: int

class GoodDealsConditionAPI(LitAPI):
    def __init__(self):
        """Initialize with pickle-safe objects only"""
        super().__init__()
        self.device = None
        self.clip_loaded = False
        self.ollama_available = False
        self.detected_object = "an item"
        self.image_cache = {}
    
    def setup(self, device):
        """Called by LitServe in each worker process"""
        self.device = "cpu"  # Force CPU for thread safety
        self._load_clip_model()
        self.ollama_available = self._test_ollama()
        self.detected_object = "an item"
    
    def _load_clip_model(self):
        """Load CLIP model safely in worker process"""
        try:
            self.model, self.preprocess = clip.load("ViT-B/32", device="cpu")
            self.model.eval()
            self.clip_loaded = True
        except Exception as e:
            print(f"Failed to load CLIP: {e}")
            self.clip_loaded = False
    
    def _test_ollama(self):
        """Test Ollama connection"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def predict(self, request: Dict[str, Any]) -> ConditionResponse:
        """Main prediction method"""
        start_time = time.time()
        
        # Parse the request dictionary
        search_query = request.get('search_query', '')
        results = request.get('results', [])
        
        # Detect object from search query (only if not already detected)
        if not hasattr(self, 'detected_object') or self.detected_object == "an item":
            object_data = self._detect_with_ollama(search_query)
            self.detected_object = object_data['detected_object']
        
        # Process all items
        enhanced_results = []
        for item in results:
            enhanced_item = self._process_item(item, {'detected_object': self.detected_object})
            enhanced_results.append(enhanced_item)
        
        return ConditionResponse(
            results=enhanced_results,
            processing_time=time.time() - start_time,
            total_processed=len(enhanced_results)
        )
    
    def encode_response(self, output: ConditionResponse) -> Dict[str, Any]:
        """LitServe calls this automatically"""
        self._save_results_to_csv(output)
        return output.model_dump()
    
    def _save_results_to_csv(self, output: ConditionResponse):
        """Save results to CSV file"""
        try:
            import pandas as pd
            import os
            from datetime import datetime
            
            os.makedirs('/app/results', exist_ok=True)
            
            results_data = []
            for item in output.results:
                results_data.append({
                    'id': item.id,
                    'title': item.title,
                    'price': item.price,
                    'source': item.source,
                    'image': item.image,
                    'location': item.location,
                    'url': item.url,
                    'miles': item.miles,
                    'detected_object': item.detected_object,
                    'condition': item.condition,
                    'condition_confidence': item.condition_confidence,
                    'processing_time': output.processing_time,
                    'total_processed': output.total_processed,
                    'timestamp': datetime.now().isoformat()
                })
            
            df = pd.DataFrame(results_data)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = f'/app/results/api_results_{timestamp}.csv'
            df.to_csv(csv_path, index=False)
            
        except Exception as e:
            print(f"Error saving CSV: {e}")
    
    def _process_item(self, item: Dict[str, Any], object_data: Dict) -> EnhancedListingItem:
        """Process single item"""
        if item.get('image') and self.clip_loaded:
            condition_data = self._assess_condition(item['image'], object_data['detected_object'])
        else:
            condition_data = {'condition': 'unknown', 'confidence': 0.0}
        
        return EnhancedListingItem(
            **item,
            detected_object=object_data['detected_object'],
            condition=condition_data['condition'],
            condition_confidence=condition_data['confidence']
        )
    
    def _assess_condition(self, image_url: str, detected_object: str) -> Dict[str, Any]:
        """Assess condition using CLIP"""
        try:
            if image_url in self.image_cache:
                image = self.image_cache[image_url]
            else:
                response = requests.get(image_url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
                self.image_cache[image_url] = image
            
            # Prepare image for CLIP
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Define condition prompts
            condition_prompts = [
                f"a {detected_object} in excellent condition",
                f"a {detected_object} in good condition", 
                f"a {detected_object} in fair condition",
                f"a {detected_object} in poor condition"
            ]
            
            # Get CLIP predictions
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(clip.tokenize(condition_prompts).to(self.device))
                
                # Calculate similarities
                similarities = (image_features @ text_features.T).softmax(dim=-1)
                condition_scores = similarities[0].cpu().numpy()
            
            # Determine condition
            conditions = ['excellent', 'good', 'fair', 'poor']
            best_idx = condition_scores.argmax()
            condition = conditions[best_idx]
            confidence = float(condition_scores[best_idx])
            
            return {'condition': condition, 'confidence': confidence}
            
        except Exception as e:
            print(f"Error assessing condition: {e}")
            return {'condition': 'unknown', 'confidence': 0.0}
    
    def _detect_with_ollama(self, search_query: str) -> Dict[str, Any]:
        """Detect object using Ollama"""
        try:
            if not self.ollama_available:
                return self._detect_fallback(search_query)
            
            prompt = f"Extract the main object from this search query: '{search_query}'. Return only the object name, nothing else."
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                detected_object = result.get('response', '').strip().lower()
                
                # Clean up the response
                if detected_object:
                    # Remove common prefixes
                    prefixes = ['the', 'a', 'an']
                    for prefix in prefixes:
                        if detected_object.startswith(prefix + ' '):
                            detected_object = detected_object[len(prefix) + 1:]
                    
                    return {'detected_object': f"a {detected_object}"}
            
            return self._detect_fallback(search_query)
            
        except Exception as e:
            print(f"Ollama detection failed: {e}")
            return self._detect_fallback(search_query)
    
    def _detect_fallback(self, search_query: str) -> Dict[str, Any]:
        """Fallback object detection"""
        query_lower = search_query.lower()
        
        # Simple keyword matching
        if 'bike' in query_lower or 'bicycle' in query_lower:
            return {'detected_object': 'a bike'}
        elif 'car' in query_lower or 'vehicle' in query_lower:
            return {'detected_object': 'a car'}
        elif 'phone' in query_lower or 'iphone' in query_lower:
            return {'detected_object': 'a phone'}
        elif 'laptop' in query_lower or 'computer' in query_lower:
            return {'detected_object': 'a laptop'}
        else:
            return {'detected_object': 'an item'}

if __name__ == "__main__":
    # Create API instance
    api = GoodDealsConditionAPI()
    
    # Start LitServe server
    server = LitServer(api, api_path="/predict")
    server.run(host="0.0.0.0", port=8000)