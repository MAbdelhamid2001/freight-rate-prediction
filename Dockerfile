# syntax=docker/dockerfile:1

# stage 1 : build stage
# Resolve the locked *runtime* dependencies (no training or dev groups) and the
# project itself into a self-contained virtualenv at /opt/venv.
FROM python:3.11-slim AS builder

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

RUN pip install --no-cache-dir uv

WORKDIR /app

# Dependency manifests first, so this layer stays cached until they change.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --no-install-project

# Then the package itself (src/freight_rate), installed non-editable.
COPY src ./src
RUN uv sync --frozen --no-default-groups --no-editable


# stage 2 : runtime stage
FROM python:3.11-slim AS runtime

# libgomp is the OpenMP runtime that LightGBM, XGBoost and scikit-learn link against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    FREIGHT_PROJECT_ROOT=/app

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser .streamlit ./.streamlit
COPY --chown=appuser:appuser models ./models
COPY --chown=appuser:appuser reports ./reports

USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

# Default: the API. docker-compose overrides the command for the Streamlit service.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
