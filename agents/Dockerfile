FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Disable pip hash checking and upgrade pip first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=20000 \
    --no-deps -r requirements.txt && \
    pip install --no-cache-dir --default-timeout=20000 -r requirements.txt

RUN python -m nltk.downloader punkt punkt_tab

COPY . .

CMD ["tail", "-f", "/dev/null"]