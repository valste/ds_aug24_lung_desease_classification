# syntax=docker/dockerfile:1.5          # ← enables --mount=type=cache
FROM python:3.11.9-slim

ENV PYTHONDONTWRITEBYTECODE=1

# ───────────── project root ──────────────
WORKDIR /app
COPY requirements.txt .

# ---------- root-only layer with caches ----------
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        dos2unix \
        libgl1 \
        libglib2.0-0 && \
    dos2unix requirements.txt && \
    pip install --upgrade pip && \
    pip install --retries 8 -r requirements.txt && \
    apt-get purge -y --auto-remove dos2unix && \
    rm -rf /var/lib/apt/lists/*
# -----------------------------------------

# non-root user (safer at runtime)
RUN adduser --disabled-password --gecos "" appuser
USER appuser

# only source code; data & models are runtime mounts
COPY --chown=appuser src ./src

# import paths work no matter where Streamlit chdir’s
ENV PYTHONPATH=/app:/app/src

EXPOSE 8501
ENTRYPOINT ["sh","-c", \
  "exec streamlit run src/streamlit/Home.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]


# --config to enable cache for pip and apt-get
# -- Windows PowerShell, Command Prompt, or Git Bash
# export DOCKER_BUILDKIT=1      # for current bash shell only
# ... and build
# docker build -t covid-xray-app .
# ------run with required mounted volumes: data + models----
# MSYS_NO_PATHCONV=1 docker run --rm -p 8501:8501 \
#   -v "$PWD":/app \
#   -v "/c/Users/User/DataScience/aug24_cds_int_analysis-of-covid-19-chest-x-rays/src/streamlit/data:/app/src/streamlit/data:ro" \
#   -v "/c/Users/User/DataScience/aug24_cds_int_analysis-of-covid-19-chest-x-rays/models:/app/models:ro" \
#   covid-xray-app