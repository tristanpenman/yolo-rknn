FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        gosu \
        libgl1 \
        libglib2.0-0 \
        tree \
        unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Which requirements file to install. The GPU service overrides this with
# python/requirements.gpu.txt (CUDA-enabled torch / onnxruntime-gpu).
ARG REQUIREMENTS=python/requirements.txt

COPY ${REQUIREMENTS} /tmp/requirements.txt

RUN python -m venv "$VIRTUAL_ENV" \
    && python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

ENV PYTHONPATH="/workspace/python"
ENV YOLO_CONFIG_DIR="/workspace/.config"

COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Creates a user/group matching the host UID/GID, then drops privileges.
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command: interactive bash shell
CMD ["bash"]
