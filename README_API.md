# ConditionAI API Server

A FastAPI-based backend service that provides AI-powered condition detection for marketplace items using CLIP and Llama 3 8B models.

## 🚀 Quick Start

### Local Development

1. **Install dependencies:**

   ```bash
   pip install -r requirements_api.txt
   ```

2. **Run the API server:**

   ```bash
   python api_server.py
   ```

3. **Test the API:**
   ```bash
   python test_api.py
   ```

The API will be available at `http://localhost:8000`

### Docker Deployment

1. **Build and run with Docker Compose:**

   ```bash
   docker-compose up --build
   ```

2. **Or build manually:**
   ```bash
   docker build -t conditionai-api .
   docker run -p 8000:8000 conditionai-api
   ```

## 🌐 Oracle Cloud Deployment

### Prerequisites

- Oracle Cloud VM instance (Ubuntu 20.04+ recommended)
- SSH access to your VM
- Docker and Docker Compose installed

### Deploy to Oracle Cloud

1. **Set environment variables:**

   ```bash
   export ORACLE_IP="your-oracle-vm-ip"
   export ORACLE_USER="ubuntu"  # or "opc" depending on your image
   export SSH_KEY="~/.ssh/your-private-key"
   ```

2. **Run deployment script:**

   ```bash
   ./deploy-oracle.sh
   ```

3. **Access your API:**
   - API: `http://your-oracle-ip:8000`
   - Health check: `http://your-oracle-ip:8000/health`
   - API docs: `http://your-oracle-ip:8000/docs`

## 📡 API Endpoints

### Health Check

```http
GET /health
```

### Process Items (Direct)

```http
POST /process-direct
Content-Type: application/json

{
  "items": [
    {
      "id": "item_001",
      "title": "Couch for sale",
      "price": "$150",
      "location_city": "Portland",
      "location_state": "OR",
      "url": "https://example.com/item_001",
      "photo_url": "https://example.com/image.jpg",
      "miles": "N/A"
    }
  ],
  "max_items": 10
}
```

### Process Items (Async)

```http
POST /process
Content-Type: application/json

{
  "items": [...],
  "max_items": 10
}
```

### Check Job Status

```http
GET /job/{job_id}
```

### Get Job Results

```http
GET /job/{job_id}/results
```

## 🔗 Next.js Integration

### 1. Environment Variables

Add to your `.env.local`:

```env
CONDITIONAI_API_URL=http://your-oracle-ip:8000
```

### 2. API Route

Create `pages/api/process-items.js` (or `app/api/process-items/route.js`):

```javascript
import { NextRequest, NextResponse } from "next/server";

const CONDITIONAI_API_URL = process.env.CONDITIONAI_API_URL;

export async function POST(request) {
  try {
    const body = await request.json();

    const response = await fetch(`${CONDITIONAI_API_URL}/process-direct`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      throw new Error("API request failed");
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
```

### 3. React Hook

Use the provided `useConditionAI` hook:

```javascript
import { useConditionAI } from "./useConditionAI";

function MyComponent() {
  const { loading, error, results, processItems } = useConditionAI();

  const handleProcess = async () => {
    try {
      const data = await processItems(items);
      console.log("Results:", data);
    } catch (err) {
      console.error("Error:", err);
    }
  };

  return (
    <div>
      <button onClick={handleProcess} disabled={loading}>
        {loading ? "Processing..." : "Analyze Conditions"}
      </button>
      {error && <div>Error: {error}</div>}
      {results && <div>Results: {JSON.stringify(results, null, 2)}</div>}
    </div>
  );
}
```

## 📊 Response Format

### Successful Response

```json
{
  "success": true,
  "results": [
    {
      "id": "item_001",
      "title": "Couch for sale",
      "price": "$150",
      "location": "Portland, OR",
      "detected_object": "a sofa",
      "object_confidence": 0.85,
      "detection_method": "llama3_8b",
      "condition": "a sofa in good condition",
      "condition_confidence": 0.92,
      "condition_2nd": "a sofa in fair condition",
      "condition_confidence_2nd": 0.75,
      "condition_3rd": "a sofa in poor condition",
      "condition_confidence_3rd": 0.23,
      "url": "https://example.com/item_001",
      "photo_url": "https://example.com/image.jpg",
      "raw_response": "sofa",
      "matched_keywords": ""
    }
  ],
  "summary": {
    "total_items": 1,
    "processed_at": "2024-01-01T12:00:00Z"
  }
}
```

## 🛠️ Configuration

### Environment Variables

- `PYTHONPATH`: Set to `/app` for Docker deployments
- `CONDITIONAI_API_URL`: Your API server URL (for Next.js)

### Model Configuration

The API automatically loads:

- CLIP ViT-L/14 model for image analysis
- Llama 3 8B model for text analysis

Models are downloaded on first run and cached locally.

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory**: Increase VM memory or use smaller batch sizes
2. **Model Download Fails**: Check internet connection and disk space
3. **API Timeout**: Increase timeout values for large batches
4. **CORS Issues**: Configure CORS origins in `api_server.py`

### Logs

Check Docker logs:

```bash
docker-compose logs -f conditionai-api
```

### Health Check

```bash
curl http://your-oracle-ip:8000/health
```

## 📈 Performance

- **Processing Time**: ~2-5 seconds per item
- **Memory Usage**: ~4-8GB RAM (depending on models)
- **Concurrent Requests**: Limited by available memory
- **Batch Size**: Recommended max 10-20 items per request

## 🔒 Security

- Configure CORS origins for production
- Use HTTPS in production
- Implement authentication if needed
- Rate limiting recommended for public APIs

## 📝 License

This project is part of the ConditionAI system. See main project for license details.
