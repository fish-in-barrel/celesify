ARG BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}
ARG INSTALL_GPU_DEPS=0

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

COPY pyproject.toml /app/pyproject.toml
COPY CLAUDE.md /app/CLAUDE.md
COPY celesify /app/celesify
RUN pip install --no-cache-dir ".[training-core,training-onnx]" && \
    if [ "$INSTALL_GPU_DEPS" = "1" ]; then pip install --no-cache-dir ".[training-gpu]"; fi

ENTRYPOINT ["python", "-m", "celesify.training.main"]
