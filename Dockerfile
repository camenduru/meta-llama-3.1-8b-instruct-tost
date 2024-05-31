FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"
RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home

RUN apt update -y && apt install software-properties-common -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg python3-pip python-is-python3

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 torchtext==0.18.0 torchdata==0.7.1 --extra-index-url https://download.pytorch.org/whl/cu121 \
    xformers==0.0.26.post1 \
    https://github.com/camenduru/wheels/releases/download/runpod/vllm-0.4.3-cp310-cp310-linux_x86_64.whl

COPY ./worker_runpod.py /content/Meta-Llama-3-8B-Instruct/worker_runpod.py
WORKDIR /content/Meta-Llama-3-8B-Instruct
CMD python worker_runpod.py
