from fastapi import APIRouter
from ing_mle_task_api.models.model import model_pipeline
from ing_mle_task_api.schemas.model_info import ModelInfoResponse, VectorizerInfoResponse
import json

router = APIRouter(tags=["Model Info"])


@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    return ModelInfoResponse(selected_model_type=model_pipeline['clf'].__class__.__name__, parameters=model_pipeline['clf'].get_params())

@router.get("/vectorizer-info", response_model=VectorizerInfoResponse)
def get_vectorizer_info():

    def custom_serializer(obj):
        if isinstance(obj, type):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    params_json = json.dumps(model_pipeline['tfidf'].get_params(), default=custom_serializer)
    params_dict = json.loads(params_json)

    return VectorizerInfoResponse(selected_vectorizer_type=model_pipeline['tfidf'].__class__.__name__, parameters=params_dict)

