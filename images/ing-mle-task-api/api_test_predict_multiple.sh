#!/bin/bash

API_URL="http://localhost:8000/predict-multiple"

PAYLOAD='{
  "titles": {
    "11": "US open: Stocks fall after Fed official hints at accelerated tapering",
    "22": "Robin Thicke writes new song dedicated to Paula Patton",
    "34": "EBay rejects Icahn slate of directors"
  }
}'

RESPONSE=$(curl -s -X POST "$API_URL" -H "Content-Type: application/json" -d "$PAYLOAD")

echo "Response from API:"
echo "$RESPONSE"