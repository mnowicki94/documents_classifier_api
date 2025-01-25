#!/bin/bash

API_URL="http://localhost:8000/predict-one"

PAYLOAD='{"title": "More apps for Android wearable devices coming soon"}'

RESPONSE=$(curl -s -X POST "$API_URL" -H "Content-Type: application/json" -d "$PAYLOAD")

echo "Response from API:"
echo "$RESPONSE"