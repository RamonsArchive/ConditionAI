from litserve import LitAPI, LitServer
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio
import aiohttp
from io import BytesIO
import time
from PIL import Image
import requests
import json
from typing import List, Optional, Union
from pydantic import BaseModel, AnyUrl
import os
import threading
import multiprocessing

# Import ML libraries
import torch
import clip

# Pydantic models for request/response
class ListingItem(BaseModel):
    id: str
    title: str
    price: Optional[Union[str, float, int]] = None
    source: Optional[str] = None
    image: Optional[str] = None          # keep as str; URLs may be non-HTTP or empty
    location: Optional[str] = None
    url: Union[str, AnyUrl, None] = None # allow plain str if not a strict URL
    miles: Optional[Union[str, float, int]] = None

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

class GoodDealsConditionAPI(LitAPI):
    def __init__(self):
        """Initialize with setup state tracking"""
        super().__init__()
        # Use simple flag instead of multiprocessing objects to avoid pickling issues
        self._setup_complete = False
        
        # Initialize all attributes to safe defaults
        self.device = "cpu"
        self.clip_loaded = False
        self.model = None
        self.preprocess = None
        self.clip_model_name = None
        self.ollama_available = False
        self.semaphore = None
    
    def setup(self, device):
        """Initialize models on server startup"""
        print("🔧 Starting setup process...")
            
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
            
        print(f"🚀 Loading CLIP model on {self.device}...")
            
        # Initialize CLIP loading flags
        self.clip_loaded = False
        self.model = None
        self.preprocess = None
        self.clip_model_name = None
            
        # Import clip at method level to avoid scope issues
        import clip
        import os
            
            # Check if we have cached models from build time
        verification_file = '/app/clip_verification.pt'
        build_time_cache = False
            
        if os.path.exists(verification_file):
            try:
                verification = torch.load(verification_file, map_location='cpu')
                print(f"✅ Found CLIP cache from build time: {verification}")
                build_time_cache = True
            except Exception as e:
                print(f"⚠️ Could not read verification file: {e}")
            
            # Try multiple approaches to load CLIP
            models_to_try = ["ViT-L/14", "ViT-B/32"]  # Primary and fallback
            
            for model_name in models_to_try:
                try:
                    print(f"📥 Attempting to load CLIP {model_name} on {self.device}...")
                    
                    # Load model with explicit error handling
                    self.model, self.preprocess = clip.load(model_name, device=self.device)
                    
                    # Verify the model loaded correctly
                    if self.model is not None and self.preprocess is not None:
                        print(f"✅ CLIP {model_name} loaded successfully!")
                        print(f"📊 Model device: {next(self.model.parameters()).device}")
                        
                        # Test the model with a simple forward pass
                        test_text = clip.tokenize(["a test item"]).to(self.device)
                        with torch.no_grad():
                            text_features = self.model.encode_text(test_text)
                            print(f"✅ CLIP model test passed, output shape: {text_features.shape}")
                        
                        self.clip_loaded = True
                        self.clip_model_name = model_name
                        print(f"🎯 Successfully loaded {model_name}")
                        break  # Success, no need to try other models
                    else:
                        raise Exception("Model or preprocess is None after loading")
                        
                except Exception as e:
                    print(f"❌ CLIP {model_name} loading failed: {e}")
                    print(f"📋 Error type: {type(e).__name__}")
                    
                    # Continue to next model
                    continue
            
            # Final check if no models loaded
            if not self.clip_loaded:
                print("❌ All CLIP model loading attempts failed!")
                try:
                    print("🔍 Available CLIP models:", clip.available_models())
                except:
                    print("🔍 Could not list available CLIP models")
                self.model = None
                self.preprocess = None
                self.clip_model_name = None
            
            # Test Ollama connection and wait for it to be ready
            print("🔍 Checking Ollama connection...")
            self.ollama_available = self._wait_for_ollama()
            if self.ollama_available:
                print("🤖 Ollama is ready!")
            else:
                print("⚠️  Ollama not available, using enhanced fallback")
            
            # Setup semaphore for concurrent processing
            self.semaphore = asyncio.Semaphore(5)  # Max 5 concurrent processes
            
            # Print final setup summary
            print("\n📋 Setup Summary:")
            print(f"   🎯 Device: {self.device}")
            print(f"   🖼️  CLIP Model: {'✅ ' + self.clip_model_name if self.clip_loaded else '❌ Failed'}")
            print(f"   🤖 Ollama: {'✅ Available' if self.ollama_available else '❌ Unavailable'}")
            print(f"   💾 Build Cache: {'✅ Found' if build_time_cache else '❌ Not found'}")
            
            if not self.clip_loaded:
                print("⚠️  WARNING: CLIP model not available - condition assessment will be limited")
            
            # Signal that setup is complete
            self._setup_complete = True
            print("🎯 Setup process completed!")
    
    def _wait_for_setup(self, timeout=60):
        """Wait for setup to complete before processing requests"""
        import time
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._setup_complete:
                    return True
            time.sleep(0.1)  # Check every 100ms
        print("❌ Setup did not complete within timeout!")
        return False
    
    def predict(self, request: ConditionRequest) -> ConditionResponse:
        """Main prediction logic with setup synchronization"""
        start_time = time.time()
        
        # Check if setup has been called at all
        print(f"🔍 Setup status: {self._setup_complete}")
        print(f"🔍 CLIP loaded: {getattr(self, 'clip_loaded', 'undefined')}")
        
        # If setup hasn't been called, try to call it manually
        if not self._setup_complete:
            print("🔧 Setup not complete, attempting to initialize...")
            try:
                self.setup("cpu")  # Force setup with CPU
            except Exception as e:
                print(f"❌ Manual setup failed: {e}")
        
        # CRITICAL: Wait for setup to complete before processing
        if not self._wait_for_setup(timeout=30):
            print("❌ Prediction called before setup completed!")
            return ConditionResponse(
                results=[],
                processing_time=time.time() - start_time,
                total_processed=0
            )
        
        try:
            print(f"📥 Received request with {len(request.results)} items")
            
            # Validate request has items
            if not request.results or len(request.results) == 0:
                print("⚠️ No items to process")
                return ConditionResponse(
                    results=[],
                    processing_time=time.time() - start_time,
                    total_processed=0
                )
            
            # Process items synchronously to avoid event loop issues
            enhanced_results = []
            for item in request.results:
                try:
                    enhanced_item = self._process_item_sync(item)
                    enhanced_results.append(enhanced_item)
                except Exception as e:
                    print(f"❌ Error processing item {item.id}: {e}")
                    # Create fallback item
                    fallback_item = EnhancedListingItem(
                        **item.model_dump(),
                        detected_object="unknown item",
                        object_confidence=0.0,
                        condition="unknown condition",
                        condition_confidence=0.0,
                        detection_method="error"
                    )
                    enhanced_results.append(fallback_item)
            
            processing_time = time.time() - start_time
            print(f"✅ Processing completed in {processing_time:.2f}s")
            
            return ConditionResponse(
                results=enhanced_results,
                processing_time=processing_time,
                total_processed=len(enhanced_results)
            )
            
        except Exception as e:
            print(f"❌ Processing error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return a safe error response instead of raising
            return ConditionResponse(
                results=[],
                processing_time=time.time() - start_time,
                total_processed=0
            )
    
    def _wait_for_ollama(self, max_attempts=30, wait_time=2) -> bool:
        """Wait for Ollama to be ready with retries"""
        import time
        for attempt in range(max_attempts):
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    # Check if our model exists
                    models = response.json().get("models", [])
                    model_names = [m.get("name", "") for m in models]
                    if any("llama3.2:1b" in name for name in model_names):
                        return True
                    else:
                        print(f"🔄 Attempt {attempt + 1}/{max_attempts}: Model not ready yet...")
                        time.sleep(wait_time)
                else:
                    print(f"🔄 Attempt {attempt + 1}/{max_attempts}: Ollama not responding...")
                    time.sleep(wait_time)
            except Exception as e:
                print(f"🔄 Attempt {attempt + 1}/{max_attempts}: Connection error: {e}")
                time.sleep(wait_time)
        return False
    
    def decode_request(self, request) -> ConditionRequest:
        """Decode incoming request"""
        print(f"📥 Raw request type: {type(request)}")
        print(f"📥 Raw request content: {request}")
        
        try:
            # Handle different request formats
            if isinstance(request, dict):
                # If request is already a dict, use it directly
                return ConditionRequest(**request)
            elif isinstance(request, list):
                # If request is a list, wrap it in the expected format
                return ConditionRequest(results=request)
            else:
                # Try to parse as JSON if it's a string
                import json
                if isinstance(request, str):
                    parsed = json.loads(request)
                    return ConditionRequest(**parsed)
                else:
                    # Fallback: assume it's the request object we want
                    return ConditionRequest(**request)
                    
        except Exception as e:
            print(f"❌ Request decoding error: {e}")
            print(f"Request content: {request}")
            # Create a fallback empty request
            return ConditionRequest(results=[])
    
    def encode_response(self, output: ConditionResponse) -> Dict[str, Any]:
        """Encode response for return"""
        return output.model_dump()
    
    def _process_item_sync(self, item: ListingItem) -> EnhancedListingItem:
        """Synchronous processing of individual item"""
        # Step 1: Detect object
        if hasattr(self, 'ollama_available') and self.ollama_available:
            object_data = self._detect_object_with_ollama_sync(item.title)
        else:
            object_data = self._detect_object_fallback(item.title)
        
        # Step 2: Assess condition
        if item.image:
            condition_data = self._assess_condition_sync(item.image, object_data['detected_object'])
        else:
            condition_data = {
                'condition': f"{object_data['detected_object']} in unknown condition",
                'condition_confidence': 0.0
            }
        
        # Combine data
        enhanced_item = EnhancedListingItem(
            **item.model_dump(),
            detected_object=object_data['detected_object'],
            object_confidence=object_data['object_confidence'],
            condition=condition_data['condition'],
            condition_confidence=condition_data['condition_confidence'],
            detection_method=object_data['detection_method']
        )
        
        return enhanced_item
    
    def _detect_object_with_ollama_sync(self, title: str) -> Dict[str, Any]:
        """Synchronous object detection using Ollama"""
        try:
            prompt = f"""You are a tagger. From the given listing title, output EXACTLY ONE generic, neutral, singular NOUN that best describes the item. 

Rules:
- Output must be ONE WORD only, lowercase letters only (^[a-z]+$). 
- No sentences, no quotes, no adjectives, no sentiment, no verbs, no punctuation.
- Ignore condition, urgency, price, free/giveaway phrasing.
- If ambiguous, output item.

Respond with the single word only to describe: {title}"""
            
            payload = {
                "model": "llama3.2:1b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 5
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                object_type = response.json()["response"].strip().lower()
                # Clean the response
                object_type = object_type.replace('.', '').replace(',', '').replace('\n', '').replace('"', '').strip()
                object_type = object_type.split()[0] if object_type.split() else "item"
                
                # Validate it's a reasonable word
                if len(object_type) > 1 and object_type.isalpha():
                    return {
                        'detected_object': f"a {object_type}",
                        'object_confidence': 0.88,
                        'detection_method': 'ollama_llama3.2_1b',
                    }
                else:
                    return self._detect_object_fallback(title)
            else:
                return self._detect_object_fallback(title)
                
        except Exception as e:
            print(f"Ollama error: {e}")
            return self._detect_object_fallback(title)
    
    def _assess_condition_sync(self, image_url: str, detected_object: str) -> Dict[str, Any]:
        """Synchronous condition assessment with better error handling"""
        try:
            # Enhanced CLIP availability check with setup synchronization
            if not self._setup_complete:
                    print("⚠️ Setup not complete, skipping condition assessment")
                    return {
                        'condition': f"{detected_object} in unknown condition (setup incomplete)",
                        'condition_confidence': 0.0
                    }
            
            if not self.clip_loaded or not self.model or not self.preprocess:
                print(f"⚠️ CLIP model not available for condition assessment")
                print(f"   clip_loaded: {getattr(self, 'clip_loaded', 'undefined')}")
                print(f"   model: {self.model is not None if hasattr(self, 'model') else 'undefined'}")
                print(f"   preprocess: {self.preprocess is not None if hasattr(self, 'preprocess') else 'undefined'}")
                
                return {
                    'condition': f"{detected_object} in unknown condition (no image analysis)",
                    'condition_confidence': 0.0
                }
            
            # Validate image URL
            if not image_url or image_url.strip() == '' or image_url.lower() == 'nan':
                print(f"⚠️ Invalid image URL: '{image_url}'")
                return {
                    'condition': f"{detected_object} in unknown condition (no image)",
                    'condition_confidence': 0.0
                }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            print(f"🔍 Downloading image for condition assessment: {image_url[:50]}...")
            # Download image with timeout
            response = requests.get(image_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            print(f"🖼️ Processing image with CLIP for {detected_object}...")
            # Process image
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Verify preprocess is callable
            if not callable(self.preprocess):
                raise Exception("Preprocess function is not callable")
                
            preprocessed_tensor = self.preprocess(image)
            image_input = preprocessed_tensor.unsqueeze(0).to(self.device)
            
            conditions = [
                f"specifically for {detected_object} in excellent condition",
                f"specifically for {detected_object} in good condition", 
                f"specifically for {detected_object} in fair condition",
                f"specifically for {detected_object} in poor condition"
            ]
            
            text = clip.tokenize(conditions).to(self.device)
            
            with torch.no_grad():
                logits_per_image, logits_per_text = self.model(image_input, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            
            results = list(zip(conditions, probs[0]))
            results.sort(key=lambda x: x[1], reverse=True)
            print(f"✅ CLIP assessment complete: {results[0][0]} (confidence: {results[0][1]:.3f})")
            
            return {
                'condition': results[0][0],
                'condition_confidence': float(results[0][1])
            }
            
        except Exception as e:
            print(f"❌ Condition assessment error: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            return {
                'condition': f"{detected_object} in unknown condition (assessment failed)",
                'condition_confidence': 0.0
            }
    
    def _detect_object_fallback(self, title: str) -> Dict[str, Any]:
        """Enhanced fallback detection with more keywords"""
        title_lower = title.lower()
        
        keyword_mapping = {
            'bicycle': ['bike', 'bicycle', 'cycling', 'trek', 'specialized', 'cannondale', 'giant', 'road bike', 'mountain bike'],
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
                # Calculate confidence based on keyword match strength
                matched_keywords = [kw for kw in keywords if kw in title_lower]
                confidence = min(0.9, 0.6 + (len(matched_keywords) * 0.1))
                return {
                    'detected_object': f"a {category}",
                    'object_confidence': confidence,
                    'detection_method': 'enhanced_keyword_fallback',
                }
        
        # If no match, try to extract likely nouns
        import re
        words = re.findall(r'\b[a-zA-Z]+\b', title_lower)
        skip_words = ['free', 'sale', 'good', 'great', 'nice', 'condition', 'used', 'new', 'excellent', 'fair', 'poor']
        
        for word in words:
            if len(word) > 3 and word not in skip_words and word.isalpha():
                # Calculate confidence based on word length and commonality
                confidence = min(0.7, 0.3 + (len(word) * 0.05))
                return {
                    'detected_object': f"a {word}",
                    'object_confidence': confidence,
                    'detection_method': 'noun_extraction',
                }
        
        return {
            'detected_object': "an item",
            'object_confidence': 0.1,  # Very low confidence for unknown items
            'detection_method': 'unknown',
        }

if __name__ == "__main__":
    # Create the API instance
    api = GoodDealsConditionAPI()
    
    # Create LitServer
    server = LitServer(
        api,
        accelerator="auto",
        max_batch_size=4,
        batch_timeout=0.1,
        workers_per_device=1,
    )
    
    # Add health check endpoint with setup status
    @server.app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "device": getattr(api, 'device', 'unknown'),
            "ollama_available": getattr(api, 'ollama_available', False),
            "clip_loaded": getattr(api, 'clip_loaded', False),
            "setup_complete": api._setup_complete,
            "models_loaded": True
        }
    
    # Add root endpoint
    @server.app.get("/")
    async def root():
        return {
            "message": "GoodDeals Condition Assessment API", 
            "status": "running",
            "setup_complete": api._setup_complete,
            "endpoints": {
                "health": "/health",
                "predict": "/predict",
                "assess_conditions": "/assess-conditions"
            }
        }
    
    # Manually add the assess-conditions endpoint that calls predict
    @server.app.post("/assess-conditions")
    async def assess_conditions(request_data: dict):
        try:
            print(f"📥 assess-conditions endpoint called with: {request_data}")
            
            # Wait for setup to complete
            if not api._wait_for_setup(timeout=30):
                return {"error": "Setup not complete", "results": [], "processing_time": 0.0, "total_processed": 0}
            
            # Call the API's predict method directly
            condition_request = api.decode_request(request_data)
            result = api.predict(condition_request)
            return api.encode_response(result)
            
        except Exception as e:
            print(f"❌ assess-conditions error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "results": [], "processing_time": 0.0, "total_processed": 0}
    
    # Add CSV processing endpoint
    @server.app.post("/process-csv")
    async def process_csv():
        """Process the listings_from_txt.csv file directly"""
        try:
            import pandas as pd
            
            print("📊 Processing listings_from_txt.csv...")
            
            # Read CSV file
            df = pd.read_csv("listings_from_txt.csv")
            print(f"📥 Loaded {len(df)} items from CSV")
            
            # Convert to ListingItem format
            items = []
            for _, row in df.iterrows():
                item = ListingItem(
                    id=str(row['id']),
                    title=str(row['title']),
                    price=str(row['price']),
                    source="csv",
                    image=str(row['photo_url']) if pd.notna(row['photo_url']) else None,
                    location=f"{row['location_city']}, {row['location_state']}",
                    url=str(row['url'])
                )
                items.append(item)
            
            # Process using the API
            condition_request = ConditionRequest(results=items)
            result = api.predict(condition_request)
            
            # Save results to CSV
            if result.results:
                results_df = pd.DataFrame([item.model_dump() for item in result.results])
                results_df.to_csv("processed_listings.csv", index=False)
                print(f"💾 Results saved to processed_listings.csv")
            
            return api.encode_response(result)
            
        except Exception as e:
            print(f"❌ CSV processing error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "results": [], "processing_time": 0.0, "total_processed": 0}
    
    print("🚀 Starting server on 0.0.0.0:8000")
    print("📋 Available endpoints:")
    print("  GET  /          - API info")
    print("  GET  /health    - Health check") 
    print("  POST /predict   - Default LitServe endpoint")
    print("  POST /assess-conditions - Custom endpoint")
    print("  POST /process-csv - Process listings_from_txt.csv directly")
    
    # Run the server
    server.run(port=8000, host="0.0.0.0")