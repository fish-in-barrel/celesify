FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY CLAUDE.md /app/CLAUDE.md
COPY celesify /app/celesify
RUN pip install --no-cache-dir ".[streamlit]"

ENTRYPOINT ["python", "-m", "streamlit", "run", "/app/celesify/streamlit_app/main.py", "--server.address=0.0.0.0", "--server.port=8501"]
