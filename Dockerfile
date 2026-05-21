FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first so pip doesn't pull the CUDA build (~8 GB → ~2 GB image)
RUN pip install --no-cache-dir \
    "torch>=2.4.0,<3.0" \
    --index-url https://download.pytorch.org/whl/cpu

# Copy project metadata first so this layer is cached separately from source
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir .

RUN mkdir -p /app/data/papers /app/data/index

ENTRYPOINT ["paper-intel"]
