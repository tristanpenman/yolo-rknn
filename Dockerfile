FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN useradd -ms /bin/bash devuser

RUN apt-get update \
    && apt-get install -y \
        build-essential \
        curl \
        git \
        unzip \
    && apt-get clean

COPY requirements.txt .

RUN pip3 install --upgrade pip \
    && pip3 install -r requirements.txt

USER devuser

WORKDIR /workspace

CMD [ "bash" ]
