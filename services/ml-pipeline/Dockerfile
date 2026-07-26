# ============ BUILD STAGE ============
FROM python:3.12-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY shared/trading-common /build/trading-common
COPY services/ml-pipeline/pyproject.toml /build/service/
# CPU-only torch: the default PyPI wheel drags in ~2.5 GB of CUDA libraries this
# service never uses (inference is a 2-layer MLP). The +cpu local version sorts
# above the plain PyPI wheel, so the resolver prefers it; everything else still
# comes from regular PyPI. If this index is ever unreachable, dropping the flag
# builds a working (just much fatter) image.
RUN pip install --no-cache-dir --prefix=/install /build/trading-common \
    && cd /build/service \
    && pip install --no-cache-dir --prefix=/install \
       --extra-index-url https://download.pytorch.org/whl/cpu .

# ============ RUNTIME STAGE ============
FROM python:3.12-slim AS runtime
COPY --from=builder /install /usr/local
WORKDIR /app
COPY services/ml-pipeline/src /app/src
RUN adduser --disabled-password --gecos "" appuser
USER appuser
EXPOSE 8000
# start-period covers the cold torch + mlflow import (~2.5 min) — without it the
# container is reported unhealthy long before it can possibly answer.
HEALTHCHECK --interval=10s --timeout=3s --retries=3 --start-period=300s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
