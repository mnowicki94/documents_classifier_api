#!/usr/bin/env bash

# Build the Docker image
docker build -t documents_classifier_api .

# Run the Docker container
docker run -p 80:80 documents_classifier_api