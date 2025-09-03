#!/bin/bash

# Oracle Cloud deployment script for ConditionAI API
# Make sure to set these environment variables:
# - ORACLE_IP: Your Oracle Cloud VM IP address
# - ORACLE_USER: Your Oracle Cloud username (usually 'ubuntu' or 'opc')
# - SSH_KEY: Path to your SSH private key

set -e

# Configuration
ORACLE_IP=${ORACLE_IP:-"your-oracle-ip"}
ORACLE_USER=${ORACLE_USER:-"ubuntu"}
SSH_KEY=${SSH_KEY:-"~/.ssh/id_rsa"}
APP_NAME="conditionai-api"
PORT=8000

echo "🚀 Deploying ConditionAI API to Oracle Cloud..."
echo "IP: $ORACLE_IP"
echo "User: $ORACLE_USER"

# Check if required variables are set
if [ "$ORACLE_IP" = "your-oracle-ip" ]; then
    echo "❌ Please set ORACLE_IP environment variable"
    exit 1
fi

# Create deployment package
echo "📦 Creating deployment package..."
tar -czf conditionai-deploy.tar.gz \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='.venv' \
    --exclude='node_modules' \
    api_server.py \
    requirements_api.txt \
    Dockerfile \
    docker-compose.yml \
    src/

# Upload to Oracle Cloud
echo "📤 Uploading to Oracle Cloud..."
scp -i "$SSH_KEY" conditionai-deploy.tar.gz "$ORACLE_USER@$ORACLE_IP:/home/$ORACLE_USER/"

# Deploy on Oracle Cloud
echo "🔧 Deploying on Oracle Cloud..."
ssh -i "$SSH_KEY" "$ORACLE_USER@$ORACLE_IP" << 'EOF'
    # Update system
    sudo apt update && sudo apt upgrade -y
    
    # Install Docker
    if ! command -v docker &> /dev/null; then
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
    fi
    
    # Install Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    fi
    
    # Create app directory
    mkdir -p /home/$USER/conditionai-api
    cd /home/$USER/conditionai-api
    
    # Extract deployment package
    tar -xzf /home/$USER/conditionai-deploy.tar.gz
    rm /home/$USER/conditionai-deploy.tar.gz
    
    # Build and start the application
    docker-compose down || true
    docker-compose build
    docker-compose up -d
    
    # Wait for the service to be ready
    echo "⏳ Waiting for service to start..."
    sleep 30
    
    # Check if service is running
    if curl -f http://localhost:8000/health; then
        echo "✅ Service is running successfully!"
    else
        echo "❌ Service failed to start"
        docker-compose logs
        exit 1
    fi
EOF

# Clean up local files
rm conditionai-deploy.tar.gz

echo "🎉 Deployment completed!"
echo "API URL: http://$ORACLE_IP:$PORT"
echo "Health check: http://$ORACLE_IP:$PORT/health"
echo "API docs: http://$ORACLE_IP:$PORT/docs"
