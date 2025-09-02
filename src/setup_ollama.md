# 🚀 Setup Guide for Llama AI Object Detection

## **Installation Steps:**

### **1. Install Ollama**

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download from https://ollama.ai/download
```

### **2. Download Llama Model**

```bash
# Download Llama 2 (7B model - good balance of speed/accuracy)
ollama pull llama2

# Or download Llama 3 (newer, better)
ollama pull llama3

# Or download Mistral (faster, smaller)
ollama pull mistral
```

### **3. Start Ollama Service**

```bash
# Start the Ollama service
ollama serve

# Keep this running in a separate terminal
```

### **4. Test the Setup**

```bash
# Test if Ollama is working
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "Hello, how are you?",
  "stream": false
}'
```

## **Update Your Code:**

In `main.py`, change the model name on line 21:

```python
'model': 'llama2',  # Change to 'llama3' or 'mistral' if you downloaded those
```

## **How It Works:**

1. **Title Input**: "Trek Super Sport Road Bike"
2. **Llama AI**: Analyzes title → Returns "bicycle"
3. **CLIP Condition**: Uses "a bicycle" for condition assessment
4. **Result**: "a bicycle in excellent condition"

## **Benefits:**

- ✅ **Free**: No API costs
- ✅ **Local**: Runs on your machine
- ✅ **Accurate**: Handles brand names and specific models
- ✅ **Fast**: ~1-2 seconds per title
- ✅ **Fallback**: Uses keyword matching if AI fails

## **Troubleshooting:**

- **Connection Error**: Make sure `ollama serve` is running
- **Model Not Found**: Run `ollama pull llama2` first
- **Slow Response**: Try `mistral` model (smaller, faster)
- **Memory Issues**: Use smaller models or increase system RAM
