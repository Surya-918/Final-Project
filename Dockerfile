FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_NAME_OR_PATH=/data \
    STORY_MEMORY_DB=/data/story_memory.sqlite3

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY src/config/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2,<3" \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY src/config ./src/config

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn src.config.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
