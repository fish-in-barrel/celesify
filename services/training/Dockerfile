ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Ensure Python + pip exist even when using a CUDA runtime base image.
RUN bash -lc 'if ! command -v python >/dev/null 2>&1; then apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && ln -sf /usr/bin/python3 /usr/bin/python; fi' \
    && bash -lc 'if ! command -v pip >/dev/null 2>&1; then ln -sf /usr/bin/pip3 /usr/bin/pip; fi' \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY train.py /app/train.py
