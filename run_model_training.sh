#!/usr/bin/env bash

# 1. Preprocess data
python src/data_preprocessing.py

# 2. Train model (saves model.pkl and tfidf_vectorizer.pkl)
python src/modeling.py
