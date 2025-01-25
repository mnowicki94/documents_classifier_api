from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from ing_mle_task_api.routers import predict, model_info

app = FastAPI(title="ING-MLE-Task-API", version="0.1.0")

app.include_router(predict.router)
app.include_router(model_info.router)

@app.get("/", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/docs")