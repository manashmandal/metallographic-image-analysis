FROM python:3.12-bullseye

WORKDIR /app

COPY pyproject.toml .
COPY poetry.lock .

RUN pip install poetry

RUN poetry config virtualenvs.create true

RUN poetry install --no-root

EXPOSE 8501

COPY . .

