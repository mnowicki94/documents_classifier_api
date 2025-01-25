FROM python:3.8.7-slim-buster

ENV POETRY_VERSION=1.8.3

ENV POETRY_HOME=/opt/poetry \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1

RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app
COPY pyproject.toml model/model.pkl model/tfidf_vectorizer.pkl ./
COPY ing_mle_task_api ./ing_mle_task_api

RUN poetry install --without dev --no-root

EXPOSE 8000

ENTRYPOINT ["poetry", "run", "uvicorn", "ing_mle_task_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
