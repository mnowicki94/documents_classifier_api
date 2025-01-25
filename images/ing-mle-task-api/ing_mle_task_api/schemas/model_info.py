from pydantic import BaseModel

class ModelInfoResponse(BaseModel):
    selected_model_type: str
    parameters: dict

class VectorizerInfoResponse(BaseModel):
    selected_vectorizer_type: str
    parameters: dict