from fastapi import FastAPI, HTTPException
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import uvicorn
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse

app = FastAPI()

# Load the model and vectorizer
model = joblib.load("models/model.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


@app.get("/", response_class=PlainTextResponse)
async def root():
    """Basic welcome message with API instructions"""
    return """Welcome to the Documents Classifier API!

Use the following endpoints:
- GET  http://0.0.0.0/predict_get?headline=your_text
- POST: via terminal:
        - python classify_headlines.py "medical procedures are about health" "robots are coming"
        - python classify_headlines.py --file titles_to_test.txt
- API docs available at /docs
"""


class HeadlineRequest(BaseModel):
    """Request model for validation"""

    headline: str


@app.get("/predict_get", summary="Predict category (GET)")
def predict_category_get(headline: str):
    """Predicts category from a given headline (GET request)"""
    if not headline.strip():
        raise HTTPException(status_code=400, detail="Headline cannot be empty.")
    X_new = tfidf_vectorizer.transform([headline])
    prediction = model.predict(X_new)[0]
    return {"category": prediction}


@app.post("/predict", summary="Predict category (POST)")
def predict_category(request: HeadlineRequest):
    """Predicts category from a given headline (POST request)"""
    X_new = tfidf_vectorizer.transform([request.headline])
    prediction = model.predict(X_new)
    return {"category": prediction[0]}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=80, reload=True)
