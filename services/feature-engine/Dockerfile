# ============ BUILD STAGE ============
FROM python:3.12-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY shared/trading-common /build/trading-common
COPY services/feature-engine/pyproject.toml /build/service/
RUN pip install --no-cache-dir --prefix=/install /build/trading-common && cd /build/service && pip install --no-cache-dir --prefix=/install .

# ============ RUNTIME STAGE ============
FROM python:3.12-slim AS runtime
COPY --from=builder /install /usr/local
WORKDIR /app
COPY services/feature-engine/src /app/src
RUN adduser --disabled-password --gecos "" appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
