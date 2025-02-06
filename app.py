from fastapi import FastAPI, HTTPException, Request, Form
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import uvicorn
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Load the model and vectorizer
model = joblib.load("models/model.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Set up templates for HTML rendering
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    """Renders the main page with the input dialog."""
    return templates.TemplateResponse("index.html", {"request": request})


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


@app.post("/predict_form", summary="Predict category (from UI)")
async def predict_category_form(headline: str = Form(...)):
    """Handles the form submission and returns prediction."""
    X_new = tfidf_vectorizer.transform([headline])
    prediction = model.predict(X_new)[0]
    return {"category": prediction}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=80, reload=True)
