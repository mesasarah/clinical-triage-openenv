# ──────────────────────────────────────────────
# Clinical Triage Navigator — OpenEnv Container
# ──────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL org.opencontainers.image.title="clinical-triage-env"
LABEL org.opencontainers.image.description="Real-world clinical triage OpenEnv environment"
LABEL org.opencontainers.image.version="1.0.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server/     ./server/
COPY graders/    ./graders/
COPY data/       ./data/
COPY openenv.yaml .
COPY inference.py .

# Make sure all packages are importable
RUN touch server/__init__.py graders/__init__.py data/__init__.py

# Switch to non-root
RUN chown -R appuser:appuser /app
USER appuser

# HF Spaces expects port 7860
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]
