FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
ENV PIP_REQUIRE_HASHES=0
RUN pip install --default-timeout=20000 --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt punkt_tab

COPY . .

CMD ["tail", "-f", "/dev/null"]