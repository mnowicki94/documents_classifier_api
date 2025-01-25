from fastapi import APIRouter
from ing_mle_task_api.models.model import model_pipeline
from ing_mle_task_api.schemas.predict import PredictOneRequest, PredictMultipleRequest, PredictMultipleResponse, PredictionResult
from ing_mle_task_api.utils.translate import translate_category

router = APIRouter(tags=["Inference"])


@router.post("/predict-one", response_model=PredictionResult)
def predict_one(request: PredictOneRequest):
    transformed_data = model_pipeline['tfidf'].transform([request.title])
    predicted_category = model_pipeline['clf'].predict(transformed_data)[0]
    predicted_label = translate_category(predicted_category)
    return PredictionResult(title=request.title, 
                           predicted_category=predicted_category, 
                           predicted_category_label=predicted_label)


@router.post("/predict-multiple", response_model=PredictMultipleResponse)
def predict_multiple(request: PredictMultipleRequest):
    predictions = {}
    for key, title in request.titles.items():
        transformed_data = model_pipeline['tfidf'].transform([title])
        predicted_category = model_pipeline['clf'].predict(transformed_data)[0]
        predicted_label = translate_category(predicted_category)
        predictions[key] = PredictionResult(title=title, predicted_category=predicted_category, predicted_category_label=predicted_label)
    return PredictMultipleResponse(predictions=predictions)
