FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Tokyo \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git git-lfs wget curl tzdata build-essential vim && \
    rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace
COPY . /workspace

ARG USERNAME=kubota
ARG UID=1000
RUN useradd -u ${UID} -m ${USERNAME} && \
    chown -R ${USERNAME}:${USERNAME} /workspace
USER ${USERNAME}

CMD ["bash"]
