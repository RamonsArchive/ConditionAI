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
        """Initialize with ONLY pickle-safe objects"""
        super().__init__()
        # DO NOT initialize models here - they can't be pickled!
        # Only simple data types that can be serialized
        self.device = None
        self.clip_loaded = False
        self.ollama_available = False
        
        # Store detected object from setup - will be set once
        self.detected_object = "an item"  # Default fallback
        
        # Simple cache - will be recreated in each worker process
        self.image_cache = {}
    
    def setup(self, device):
        """Called by LitServer in EACH worker process - models are safe here"""
        import os
        print(f"Worker process {os.getpid()}: Setting up models...")
        
        # This runs in worker process - safe to load models
        self.device = "cpu"  # Force CPU for thread safety
        
        # Load CLIP model (happens in worker process)
        self._load_clip_model()
        
        # Check Ollama (happens in worker process)  
        self.ollama_available = self._test_ollama()
        
        # Set default detected object (will be overridden by first request)
        self.detected_object = "an item"
        
        print(f"Worker {os.getpid()}: CLIP={self.clip_loaded}, Ollama={self.ollama_available}")
    
    def _load_clip_model(self):
        """Load CLIP safely in worker process"""
        try:
            print("Loading CLIP model in worker process...")
            
            # Load directly without storing complex objects during init
            self.model, self.preprocess = clip.load("ViT-B/32", device="cpu")
            self.model.eval()
            
            # Test it works
            test_text = clip.tokenize(["test"]).to("cpu")
            with torch.no_grad():
                _ = self.model.encode_text(test_text)
            
            self.clip_loaded = True
            print("CLIP loaded successfully in worker process")
            
        except Exception as e:
            print(f"CLIP loading failed in worker: {e}")
            self.clip_loaded = False
            self.model = None
            self.preprocess = None
    
    def _test_ollama(self):
        """Test Ollama connection in worker process"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def decode_request(self, request) -> ConditionRequest:
        """LitServer calls this automatically"""
        if isinstance(request, dict):
            return ConditionRequest(**request)
        return ConditionRequest(search_query="", results=[])
    
    def predict(self, request: ConditionRequest) -> ConditionResponse:
        """Main processing - runs in worker process"""
        start_time = time.time()
        
        if not request.results:
            return ConditionResponse(results=[], processing_time=0, total_processed=0)
        
        # Check if setup has been called (for single worker mode)
        if not hasattr(self, 'model') or self.model is None:
            print("🔧 Setup not called yet - calling setup manually...")
            self.setup("cpu")
        
        # Object detection - only if we don't have one stored yet
        if self.detected_object == "an item" and request.search_query:
            print(f"🔍 First request - detecting object from: {request.search_query}")
            if self.ollama_available:
                object_data = self._detect_with_ollama(request.search_query)
            else:
                object_data = self._detect_fallback(request.search_query)
            
            # Store for future requests
            self.detected_object = object_data['detected_object']
            print(f"✅ Stored detected object: {self.detected_object}")
        else:
            print(f"♻️ Reusing stored object: {self.detected_object}")
            object_data = {'detected_object': self.detected_object}
        
        # Process items
        enhanced_results = []
        for item in request.results:
            enhanced_item = self._process_item(item, object_data)
            enhanced_results.append(enhanced_item)
        
        return ConditionResponse(
            results=enhanced_results,
            processing_time=time.time() - start_time,
            total_processed=len(enhanced_results)
        )
    
    def encode_response(self, output: ConditionResponse) -> Dict[str, Any]:
        """LitServer calls this automatically"""
        # Save results to CSV file
        self._save_results_to_csv(output)
        return output.model_dump()
    
    def _save_results_to_csv(self, output: ConditionResponse):
        """Save results to CSV file for local extraction"""
        try:
            import pandas as pd
            import os
            from datetime import datetime
            
            # Create results directory if it doesn't exist
            os.makedirs('/app/results', exist_ok=True)
            
            # Convert results to DataFrame
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
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(results_data)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = f'/app/results/api_results_{timestamp}.csv'
            df.to_csv(csv_path, index=False)
            
            print(f"💾 Results saved to: {csv_path}")
            
        except Exception as e:
            print(f"❌ Error saving results to CSV: {e}")
    
    def _process_item(self, item: ListingItem, object_data: Dict) -> EnhancedListingItem:
        """Process single item"""
        if item.image and self.clip_loaded:
            condition_data = self._assess_condition(item.image, object_data['detected_object'])
        else:
            condition_data = {'condition': "unknown", 'condition_confidence': 0.0}
        
        return EnhancedListingItem(
            **item.model_dump(),
            detected_object=object_data['detected_object'],
            condition=condition_data['condition'],
            condition_confidence=condition_data['condition_confidence']
        )
    
    def _assess_condition(self, image_url: str, detected_object: str) -> Dict:
        """Assess condition with CLIP"""
        try:
            # Download image
            if image_url in self.image_cache:
                image_data = self.image_cache[image_url]
            else:
                response = requests.get(image_url, timeout=10)
                image_data = response.content
                self.image_cache[image_url] = image_data
            
            # Process with CLIP
            image = Image.open(BytesIO(image_data)).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to("cpu")
            
            conditions = [
                f"{detected_object} in excellent condition",
                f"{detected_object} in good condition", 
                f"{detected_object} in fair condition",
                f"{detected_object} in poor condition"
            ]
            
            text = clip.tokenize(conditions).to("cpu")
            
            with torch.no_grad():
                logits_per_image, _ = self.model(image_input, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            
            best_idx = probs[0].argmax()
            condition = conditions[best_idx].split(" in ")[1].split(" condition")[0]
            confidence = float(probs[0][best_idx])
            
            return {'condition': condition, 'condition_confidence': round(confidence, 3)}
            
        except Exception as e:
            print(f"Condition assessment error: {e}")
            return {'condition': "unknown", 'condition_confidence': 0.0}
    
    def _detect_with_ollama(self, search_query: str) -> Dict:
        """Object detection with Ollama"""
        try:
            payload = {
                "model": "llama3.2:1b",
                "prompt": f"What object is being searched for? One word only. Example bike in san diego: bike. Example vintage bicycle for sale: bicycle. Example used car near me: car. Output one word only: {search_query}",
                "stream": False
            }
            
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=15)
            
            if response.status_code == 200:
                object_type = response.json()["response"].strip().lower()
                object_type = object_type.replace('.', '').replace(',', '').strip()
                if object_type and object_type.isalpha():
                    return {'detected_object': f"a {object_type}"}
            
        except Exception as e:
            print(f"Ollama error: {e}")
        
        return self._detect_fallback(search_query)
    
    def _detect_fallback(self, search_query: str) -> Dict:
        """Simple keyword-based detection"""
        query_lower = search_query.lower()
        
        if 'bike' in query_lower or 'bicycle' in query_lower:
            return {'detected_object': "a bicycle"}
        elif 'car' in query_lower or 'vehicle' in query_lower:
            return {'detected_object': "a car"}
        elif 'phone' in query_lower:
            return {'detected_object': "a phone"}
        else:
            return {'detected_object': "an item"}


# CRITICAL: Clean main section that avoids pickle issues
if __name__ == "__main__":
    import os
    
    # Create API instance with NO model loading
    api = GoodDealsConditionAPI()
    
    # DO NOT call setup() here - it will cause pickle errors!
    # LitServer will call setup() in each worker process
    
    # Create server
    server = LitServer(
        api,
        accelerator="cpu",  # Avoid CUDA issues
        workers_per_device=1,  # Single worker
        timeout=60
    )
    
    @server.app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    print("Starting LitServer...")
    print("Models will be loaded automatically in worker processes")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /predict - Default LitServe endpoint")
    
    # This will work without pickle errors
    server.run(port=8000, host="0.0.0.0")