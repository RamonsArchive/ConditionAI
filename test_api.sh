#!/bin/bash

# Test script for Condition AI API
# Tests the /predict endpoint with all 54 items from listings_from_txt.csv

echo "🚀 Testing Condition AI API with all 54 items..."

# Check if test_data.json exists
if [ ! -f "test_data.json" ]; then
    echo "❌ test_data.json not found. Please run convert_csv_to_json.py first."
    exit 1
fi

# Test the predict endpoint with all items
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_data.json

echo ""
echo "✅ Test completed with all 54 items!"
