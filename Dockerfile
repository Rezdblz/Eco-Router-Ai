FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir \
    httpx \
    python-dotenv \
    orjson \
    pydantic \
    pytest

RUN mkdir -p /output

CMD ["python", "-u", "app/main.py"]
