# ConditionAI API

A machine learning API for assessing the condition of items from marketplace listings using CLIP and Ollama.

## Files

- `railway_main.py` - Local development version
- `railway_main_server.py` - Server/Docker version (identical to local)
- `Dockerfile` - Docker configuration for local development
- `Dockerfile_server.py` - Docker configuration for server deployment
- `preload_clip.py` - Script to pre-load CLIP models at build time
- `requirements.txt` - Python dependencies
- `listings_from_txt.csv` - Sample input data
- `test_assess_conditions.sh` - Test script for API endpoints

## API Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `POST /assess-conditions` - Assess condition of items
- `POST /process-csv` - Process CSV file directly

## Local Development

```bash
python railway_main.py
```

## Docker Development

```bash
docker build -t condition-ai .
docker run -p 8000:8000 condition-ai
```

## Server Deployment

Use `railway_main_server.py` and `Dockerfile_server.py` for production deployment.

## Testing

```bash
# Test health
curl http://localhost:8000/health

# Test assess-conditions
bash test_assess_conditions.sh
```
