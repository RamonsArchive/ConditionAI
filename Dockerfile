# Condition AI API Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY condition_ai.py .
COPY preload_clip.py .

# Create necessary directories
RUN mkdir -p /app/torch_cache /app/clip_cache /app/results

# Preload CLIP model (with error handling)
RUN python preload_clip.py || echo "CLIP preloading failed, will load at runtime"

# Verify CLIP cache (optional)
RUN python -c "import os; print('Verifying CLIP cache...'); import torch; data = torch.load('/app/clip_verification.pt'); print('CLIP verification loaded successfully:', data)" || echo "CLIP verification file not found, will create at runtime"

# Create startup script
RUN echo '#!/bin/bash\n\
    set -e\n\
    echo "🚀 Starting Ollama service..."\n\
    ollama serve &\n\
    OLLAMA_PID=$!\n\
    echo "⏳ Waiting for Ollama to start..."\n\
    sleep 10\n\
    echo "📥 Downloading Llama model..."\n\
    ollama pull llama3.2:3b || echo "Llama model pull failed, continuing..."\n\
    echo "✅ Ollama ready!"\n\
    echo "🚀 Starting Condition AI API..."\n\
    python condition_ai.py\n\
    ' > start.sh && chmod +x start.sh

# Expose port
EXPOSE 8000

# Start the application
CMD ["./start.sh"]