FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=20000 \
    --no-deps -r requirements.txt && \
    pip install --no-cache-dir --default-timeout=20000 -r requirements.txt

RUN python -m spacy download it_core_news_sm

COPY src/ .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "server_api.py", "--host", "0.0.0.0", "--port", "8000"]