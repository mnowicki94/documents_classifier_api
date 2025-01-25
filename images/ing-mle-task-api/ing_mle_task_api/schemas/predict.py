from pydantic import BaseModel
from typing import Dict

class PredictOneRequest(BaseModel):
    title: str

class PredictionResult(BaseModel):
    title: str
    predicted_category: str
    predicted_category_label: str

class PredictMultipleRequest(BaseModel):
    titles: Dict[str, str]

class PredictMultipleResponse(BaseModel):
    predictions: Dict[str, PredictionResult]
