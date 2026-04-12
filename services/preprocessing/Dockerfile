FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY CLAUDE.md /app/CLAUDE.md
COPY celesify /app/celesify
RUN pip install --no-cache-dir ".[preprocessing]"

ENTRYPOINT ["python", "-m", "celesify.preprocessing.main"]
