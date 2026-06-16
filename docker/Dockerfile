FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r agentuser && useradd -r -g agentuser -d /app agentuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/data /app/models \
    && chown -R agentuser:agentuser /app

USER agentuser

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
  CMD curl -f http://localhost:8003/health || exit 1

ENV PYTHONPATH=/app
ENV DATA_DIR=/app/data
ENV MODELS_DIR=/app/models

ENTRYPOINT ["python", "agent/main.py"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8003"]
