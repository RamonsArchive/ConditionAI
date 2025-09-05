FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create CLIP cache directory and set environment variables
ENV TORCH_HOME=/app/torch_cache
ENV CLIP_CACHE=/app/clip_cache
RUN mkdir -p $TORCH_HOME $CLIP_CACHE

# Pre-load CLIP model at build time for faster startup
# This downloads and caches the model files
RUN python preload_clip.py

# Verify the model was cached - FIXED SYNTAX
RUN python -c "import os; print('Verifying CLIP cache...'); import torch; data = torch.load('/app/clip_verification.pt', map_location='cpu'); print('CLIP cache verified:', data)"

# Create startup script that starts Ollama and pulls model - FIXED F-STRING
RUN echo '#!/bin/bash\n\
set -e\n\
echo "🚀 Starting Ollama service..."\n\
ollama serve &\n\
OLLAMA_PID=$!\n\
echo "⏳ Waiting for Ollama to be ready..."\n\
sleep 10\n\
echo "📥 Pulling llama3.2:1b model..."\n\
ollama pull llama3.2:1b || echo "⚠️ Failed to pull model, continuing..."\n\
echo "🎯 Verifying CLIP model cache..."\n\
python -c "import torch; data = torch.load(\\"/app/clip_verification.pt\\", map_location=\\"cpu\\"); print(\\"CLIP cache exists:\\", data)" || echo "❌ CLIP verification failed"\n\
echo "🚀 Starting application..."\n\
python railway_main.py\n\
' > start.sh && chmod +x start.sh

# Expose port
EXPOSE 8000

# Run the startup script
CMD ["./start.sh"]