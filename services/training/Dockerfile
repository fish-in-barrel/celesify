ARG BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Make the image resilient when the base image does not provide Python tooling.
RUN set -eux; \
    if ! command -v python >/dev/null 2>&1; then \
        apt-get update; \
        apt-get install -y --no-install-recommends python3 python3-pip; \
        ln -sf /usr/bin/python3 /usr/bin/python; \
    fi; \
    if ! command -v pip >/dev/null 2>&1; then \
        ln -sf /usr/bin/pip3 /usr/bin/pip; \
    fi; \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY train.py /app/train.py
