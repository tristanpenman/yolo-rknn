# check=skip=FromPlatformFlagConstDisallowed

FROM --platform=linux/amd64 python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt

RUN python -m venv "$VIRTUAL_ENV" \
    && python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt
