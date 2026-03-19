# =============================================================================
# Dockerfile — doge_predictor inference server
#
# Multi-stage build:
#   Stage 1 (builder)  — compile TA-Lib C library from source, install Python deps
#   Stage 2 (runtime)  — slim image; copy compiled artefacts + app code
#
# Exposed ports:
#   8000 — GET /health        (HealthCheckServer)
#   8001 — GET /metrics       (Prometheus)
#   8080 — Trading dashboard  (FastAPI + TradingView Lightweight Charts)
#
# Health check:
#   curl -f http://localhost:8000/health || exit 1
#
# Usage:
#   docker build -t doge_predictor:latest .
#   docker run -p 8000:8000 -p 8001:8001 -p 8080:8080 \
#     -e BINANCE_API_KEY=... -e BINANCE_API_SECRET=... \
#     -v $(pwd)/models:/app/models \
#     doge_predictor:latest
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — builder: compile TA-Lib C library + install all Python deps
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Install OS build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        ca-certificates \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ----- Build TA-Lib C library from source -----------------------------------
# ta-lib>=0.5 bundles the C lib on Windows wheels but Linux pip installs still
# require the shared library to be present at runtime.  Build from source for
# maximum portability and reproducibility.
ARG TALIB_VERSION=0.4.0
RUN wget -q "https://sourceforge.net/projects/ta-lib/files/ta-lib/${TALIB_VERSION}/ta-lib-${TALIB_VERSION}-src.tar.gz" \
    && tar xzf "ta-lib-${TALIB_VERSION}-src.tar.gz" \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib "ta-lib-${TALIB_VERSION}-src.tar.gz"

# ----- Install Python dependencies ------------------------------------------
WORKDIR /build

# Copy only requirements first (layer cache friendly)
COPY requirements.txt ./

# Install all Python dependencies into /build/venv
RUN python -m venv /build/venv
RUN /build/venv/bin/pip install --upgrade pip wheel setuptools
RUN /build/venv/bin/pip install --no-cache-dir -r requirements.txt


# ---------------------------------------------------------------------------
# Stage 2 — runtime: slim image with only what we need
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL maintainer="doge_predictor" \
      description="DOGE price prediction inference server" \
      version="1.0"

# Install only runtime OS dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy TA-Lib shared library from builder stage
COPY --from=builder /usr/lib/libta_lib* /usr/lib/
COPY --from=builder /usr/lib/libta-lib* /usr/lib/
RUN ldconfig

# Copy Python virtual environment from builder stage
COPY --from=builder /build/venv /app/venv

# Set PATH so the venv python is used by default
ENV PATH="/app/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app"

# ---------------------------------------------------------------------------
# Application code
# ---------------------------------------------------------------------------
WORKDIR /app

# Copy the full project (excluding items in .dockerignore)
COPY . .

# Create directories that must exist at runtime
RUN mkdir -p logs models data

# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------
EXPOSE 8000
EXPOSE 8001
EXPOSE 8080

# ---------------------------------------------------------------------------
# Health check (relies on curl installed above)
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ---------------------------------------------------------------------------
# Default command
# ---------------------------------------------------------------------------
CMD ["python", "scripts/serve.py", \
     "--models-dir", "models", \
     "--health-port", "8000", \
     "--metrics-port", "8001"]
