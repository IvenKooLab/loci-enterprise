# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# No runtime dependencies, so there is nothing to resolve — the image is the
# package plus a user and a data directory.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . \
 && useradd --create-home --uid 10001 loci \
 && mkdir -p /data && chown loci:loci /data

USER loci
VOLUME ["/data"]
ENV LE_DB=/data/le.db

# 8971 is the HTTP API; 127.0.0.1 is deliberately not used here — inside a
# container the service must bind all interfaces to be reachable.
EXPOSE 8971
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8971/health',timeout=2).status==200 else 1)"

ENTRYPOINT ["le", "--db", "/data/le.db"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8971"]
