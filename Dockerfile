# ── Stage 1: build ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch pre-installed so sentence-transformers never pulls CUDA build
RUN pip install --no-cache-dir \
    "torch>=2.4.0,<3.0" \
    --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir .

# Strip runtime-unnecessary packages HERE in the builder stage.
# Deleting before the COPY to the final stage means the final image
# never contains these files — unlike deleting in the final stage where
# the files still exist in the lower COPY layer.
RUN rm -rf \
    /usr/local/lib/python3.11/site-packages/torch/test \
    /usr/local/lib/python3.11/site-packages/torch/include \
    /usr/local/lib/python3.11/site-packages/pyarrow \
    /usr/local/lib/python3.11/site-packages/pyarrow*.dist-info \
    /usr/local/lib/python3.11/site-packages/pandas \
    /usr/local/lib/python3.11/site-packages/pandas-*.dist-info \
    /usr/local/lib/python3.11/site-packages/pydeck \
    /usr/local/lib/python3.11/site-packages/pydeck-*.dist-info \
    /usr/local/lib/python3.11/site-packages/streamlit \
    /usr/local/lib/python3.11/site-packages/streamlit-*.dist-info \
    /usr/local/lib/python3.11/site-packages/altair \
    /usr/local/lib/python3.11/site-packages/altair-*.dist-info \
    /usr/local/lib/python3.11/site-packages/gitpython \
    /usr/local/lib/python3.11/site-packages/gitpython-*.dist-info \
    /usr/local/lib/python3.11/site-packages/GitPython-*.dist-info \
    /usr/local/lib/python3.11/site-packages/watchdog \
    /usr/local/lib/python3.11/site-packages/watchdog-*.dist-info

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
# Fresh python:3.11-slim with no compiler tools — 277 MB lighter than single-stage.
FROM python:3.11-slim

WORKDIR /app

# Only the already-stripped site-packages and the CLI entry point come across.
# gcc/g++ never exists in this stage.
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
# Copy all pip-installed scripts (uvicorn, paper-intel, huggingface-cli, etc.)
COPY --from=builder /usr/local/bin/ /usr/local/bin/

RUN mkdir -p /app/data/papers /app/data/index

ENTRYPOINT ["paper-intel"]
