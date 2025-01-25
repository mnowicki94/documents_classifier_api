# (A) Running from python virtual environment (development)

## Prerequisities 
Python 3.8.7 and poetry (1.8.3) installed

## 1. Create virtual environment using pyproject.toml

    poetry install


## 2.Run API service:

    poetry run uvicorn ing_mle_task_api.main:app --reload

# (B) Running in a container

## Prerequisities 
Docker installed

## 1. Build docker image:

    docker build -t ingmletaskapi:latest .

## 2. Run container from docker image:

    docker run -d -p 8000:8000 ingmletaskapi:latest

## 3. API service should be available under:

    http://localhost:8000/


# Testing

## Bash Script to Test API (api_test.sh)

### /predict-one

```bash
#!/bin/bash

API_URL="http://localhost:8000/predict-one"

PAYLOAD='{"title": "More apps for Android wearable devices coming soon"}'

RESPONSE=$(curl -s -X POST "$API_URL" -H "Content-Type: application/json" -d "$PAYLOAD")

echo "Response from API:"
echo "$RESPONSE"
```

### /predict-multiple

```bash
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
```

## pytest

To run all tests included in repository run:

    pytest