# Condition AI API

A high-performance API for assessing the condition of items in marketplace listings using computer vision and natural language processing.

## Features

- **CLIP-based condition assessment** - Uses OpenAI's CLIP model for image analysis
- **Ollama integration** - Local LLM for object detection from search queries
- **LitServe framework** - High-performance serving with multiprocessing
- **Automatic CSV export** - Results saved to timestamped CSV files
- **Docker ready** - Containerized for easy deployment

## Quick Start

### 1. Build and Run with Docker

```bash
# Build the image
docker build -t condition-ai .

# Run the container
docker run -d -p 8000:8000 --name condition-ai condition-ai
```

### 2. Test the API

```bash
# Make the test script executable
chmod +x test_api.sh

# Run the test
./test_api.sh
```

### 3. API Usage

**Endpoint:** `POST /predict`

**Request:**

```json
{
  "search_query": "bicycle in san diego",
  "results": [
    {
      "id": "123",
      "title": "Mountain Bike",
      "price": "$100",
      "source": "facebook",
      "image": "https://example.com/image.jpg",
      "location": "San Diego, CA",
      "url": "https://example.com/item",
      "miles": null
    }
  ]
}
```

**Response:**

```json
{
  "results": [
    {
      "id": "123",
      "title": "Mountain Bike",
      "price": "$100",
      "source": "facebook",
      "image": "https://example.com/image.jpg",
      "location": "San Diego, CA",
      "url": "https://example.com/item",
      "miles": null,
      "detected_object": "a bike",
      "condition": "good",
      "condition_confidence": 0.85
    }
  ],
  "processing_time": 2.5,
  "total_processed": 1
}
```

## Performance

- **Processing time**: ~2-15 seconds per request (depending on image count)
- **CSV export overhead**: <0.1% of total processing time
- **Concurrent requests**: Supports multiple workers via LitServe
- **Memory usage**: ~2GB RAM (includes CLIP model)

## Files

- `condition_ai.py` - Main API application
- `preload_clip.py` - CLIP model preloading script
- `test_api.sh` - API testing script
- `listings_from_txt.csv` - Sample data
- `Dockerfile` - Container configuration
- `requirements.txt` - Python dependencies

## Next Steps for Production

1. **Deploy to cloud** (AWS, GCP, Azure, Railway, etc.)
2. **Add authentication** (API keys, JWT tokens)
3. **Implement rate limiting**
4. **Add monitoring** (Prometheus, Grafana)
5. **Scale horizontally** (multiple containers)
6. **Add caching** (Redis for image cache)

## Next.js Integration

To integrate with Next.js:

```javascript
// API call from Next.js
const response = await fetch("https://your-api-domain.com/predict", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    search_query: searchQuery,
    results: listings,
  }),
});

const data = await response.json();
```

## License

MIT License
