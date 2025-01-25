from fastapi import FastAPI
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import uvicorn

app = FastAPI()

model = joblib.load("models/model.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


@app.get("/")
def read_root():
    return {"message": "Welcome to my FastAPI app!"}


# Add a GET-based route for quick browser testing
@app.get("/predict_get")
def predict_category_get(headline: str):
    # Example: http://0.0.0.0:8000/predict_get?headline=some+text
    X_new = tfidf_vectorizer.transform([headline])
    prediction = model.predict(X_new)[0]
    return {"category": prediction}


@app.post("/predict")
def predict_category(request: dict):
    text = request.get("headline", "")
    X_new = tfidf_vectorizer.transform([text])
    prediction = model.predict(X_new)
    return {"category": prediction[0]}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
