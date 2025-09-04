FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create startup script that starts Ollama and pulls model
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting Ollama service..."\n\
ollama serve &\n\
OLLAMA_PID=$!\n\
sleep 10\n\
echo "Pulling llama3.2:1b model..."\n\
ollama pull llama3.2:1b\n\
echo "Model downloaded, starting application..."\n\
python railway_main.py\n\
' > start.sh && chmod +x start.sh

# Expose port
EXPOSE 8000

# Run the startup script
CMD ["./start.sh"]