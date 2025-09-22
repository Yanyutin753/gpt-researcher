# Backend-only, fast and stable Dockerfile (no browsers)

FROM python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_ROOT_USER_ACTION=ignore \
    UVICORN_WORKERS=1

WORKDIR /usr/src/app

# System dependencies for building wheels and runtime libs
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    gcc g++ \
    rustc cargo \
    git \
    wget curl ca-certificates \
    pkg-config \
    libxml2-dev libxslt1-dev \
    libjpeg-dev zlib1g-dev \
    libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf-2.0-0 \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Separate layer for Python deps to leverage Docker cache
COPY ./requirements.txt ./requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefer-binary -r requirements.txt

# Create non-root user and prepare writable dirs
RUN useradd -ms /bin/bash gpt-researcher \
    && mkdir -p /usr/src/app/outputs \
    && chown -R gpt-researcher:gpt-researcher /usr/src/app

USER gpt-researcher

# Copy source
COPY --chown=gpt-researcher:gpt-researcher ./ ./

# Runtime configuration
ARG HOST=0.0.0.0
ENV HOST=${HOST}
ARG PORT=8000
ENV PORT=${PORT}
ARG WORKERS=1
ENV WORKERS=${WORKERS}
EXPOSE ${PORT}

# Lightweight healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
    CMD wget -qO- http://127.0.0.1:${PORT}/v1/models > /dev/null || exit 1

CMD ["sh","-c","uvicorn main:app --host ${HOST} --port ${PORT} --workers ${WORKERS} --proxy-headers --forwarded-allow-ips '*'" ]
