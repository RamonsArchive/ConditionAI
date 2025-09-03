from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import json
import asyncio
from datetime import datetime
import uuid
import os

# Import your existing functions
from src.main import process_csv_file, detect_object_with_llama, assess_condition

app = FastAPI(
    title="ConditionAI API",
    description="AI-powered condition detection for marketplace items",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ItemData(BaseModel):
    id: str
    title: str
    price: str
    location_city: str
    location_state: str
    url: str
    photo_url: str
    miles: Optional[str] = "N/A"

class ProcessRequest(BaseModel):
    items: List[ItemData]
    max_items: Optional[int] = None

class ProcessResponse(BaseModel):
    job_id: str
    status: str
    message: str
    total_items: int

class ItemResult(BaseModel):
    id: str
    title: str
    price: str
    location: str
    detected_object: str
    object_confidence: float
    detection_method: str
    condition: str
    condition_confidence: float
    condition_2nd: str
    condition_confidence_2nd: float
    condition_3rd: str
    condition_confidence_3rd: float
    url: str
    photo_url: str
    raw_response: Optional[str] = None
    matched_keywords: Optional[str] = None

class JobStatus(BaseModel):
    job_id: str
    status: str  # "processing", "completed", "failed"
    progress: int  # 0-100
    total_items: int
    processed_items: int
    results: Optional[List[ItemResult]] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

# In-memory storage for job status (use Redis/DB in production)
job_storage: Dict[str, JobStatus] = {}

@app.get("/")
async def root():
    return {"message": "ConditionAI API is running!", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.post("/process", response_model=ProcessResponse)
async def process_items(request: ProcessRequest, background_tasks: BackgroundTasks):
    """Start processing items for condition detection"""
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create job status
        job_status = JobStatus(
            job_id=job_id,
            status="processing",
            progress=0,
            total_items=len(request.items),
            processed_items=0,
            created_at=datetime.now()
        )
        job_storage[job_id] = job_status
        
        # Start background processing
        background_tasks.add_task(process_items_background, job_id, request.items, request.max_items)
        
        return ProcessResponse(
            job_id=job_id,
            status="processing",
            message="Processing started",
            total_items=len(request.items)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting processing: {str(e)}")

async def process_items_background(job_id: str, items: List[ItemData], max_items: Optional[int]):
    """Background task to process items"""
    try:
        job_status = job_storage[job_id]
        
        # Convert to DataFrame for processing
        items_data = [item.dict() for item in items]
        df = pd.DataFrame(items_data)
        
        # Save temporary CSV
        temp_csv = f"temp_{job_id}.csv"
        df.to_csv(temp_csv, index=False)
        
        # Process items
        results = process_csv_file(temp_csv, max_items=max_items)
        
        # Convert results to ItemResult objects
        item_results = []
        for result in results:
            item_result = ItemResult(
                id=result['id'],
                title=result['title'],
                price=result['price'],
                location=result['location'],
                detected_object=result['detected_object'],
                object_confidence=result['object_confidence'],
                detection_method=result['detection_method'],
                condition=result['condition'],
                condition_confidence=result['condition_confidence'],
                condition_2nd=result['condition_2nd'],
                condition_confidence_2nd=result['condition_confidence_2nd'],
                condition_3rd=result['condition_3rd'],
                condition_confidence_3rd=result['condition_confidence_3rd'],
                url=result['url'],
                photo_url=result['photo_url'],
                raw_response=result.get('raw_response'),
                matched_keywords=result.get('matched_keywords')
            )
            item_results.append(item_result)
        
        # Update job status
        job_status.status = "completed"
        job_status.progress = 100
        job_status.processed_items = len(item_results)
        job_status.results = item_results
        job_status.completed_at = datetime.now()
        
        # Clean up temp file
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
            
    except Exception as e:
        # Update job status with error
        job_status = job_storage[job_id]
        job_status.status = "failed"
        job_status.error_message = str(e)
        job_status.completed_at = datetime.now()
        
        # Clean up temp file
        temp_csv = f"temp_{job_id}.csv"
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

@app.get("/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a processing job"""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_storage[job_id]

@app.get("/job/{job_id}/results")
async def get_job_results(job_id: str):
    """Get the results of a completed job"""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_status = job_storage[job_id]
    
    if job_status.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    return {
        "job_id": job_id,
        "results": job_status.results,
        "summary": {
            "total_items": job_status.total_items,
            "processed_items": job_status.processed_items,
            "completed_at": job_status.completed_at
        }
    }

@app.delete("/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its results"""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del job_storage[job_id]
    return {"message": "Job deleted successfully"}

@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job.status,
                "progress": job.progress,
                "total_items": job.total_items,
                "created_at": job.created_at,
                "completed_at": job.completed_at
            }
            for job_id, job in job_storage.items()
        ]
    }

# Direct processing endpoint (synchronous)
@app.post("/process-direct")
async def process_items_direct(request: ProcessRequest):
    """Process items directly and return results (for small batches)"""
    try:
        # Convert to DataFrame
        items_data = [item.dict() for item in request.items]
        df = pd.DataFrame(items_data)
        
        # Save temporary CSV
        temp_csv = f"temp_direct_{uuid.uuid4()}.csv"
        df.to_csv(temp_csv, index=False)
        
        # Process items
        results = process_csv_file(temp_csv, max_items=request.max_items)
        
        # Convert results
        item_results = []
        for result in results:
            item_result = ItemResult(
                id=result['id'],
                title=result['title'],
                price=result['price'],
                location=result['location'],
                detected_object=result['detected_object'],
                object_confidence=result['object_confidence'],
                detection_method=result['detection_method'],
                condition=result['condition'],
                condition_confidence=result['condition_confidence'],
                condition_2nd=result['condition_2nd'],
                condition_confidence_2nd=result['condition_confidence_2nd'],
                condition_3rd=result['condition_3rd'],
                condition_confidence_3rd=result['condition_confidence_3rd'],
                url=result['url'],
                photo_url=result['photo_url'],
                raw_response=result.get('raw_response'),
                matched_keywords=result.get('matched_keywords')
            )
            item_results.append(item_result)
        
        # Clean up
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
        
        return {
            "results": item_results,
            "summary": {
                "total_items": len(item_results),
                "processed_at": datetime.now()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing items: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
