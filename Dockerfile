FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"
RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home

RUN apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av xformers==0.0.25 runpod \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiTRUE-cp310-cp310-linux_x86_64.whl \
    https://github.com/camenduru/wheels/releases/download/tost/vllm-0.4.2-cp310-cp310-linux_x86_64.whl

COPY ./worker_runpod.py /content/Meta-Llama-3-8B-Instruct/worker_runpod.py
WORKDIR /content/Meta-Llama-3-8B-Instruct
CMD python worker_runpod.py
