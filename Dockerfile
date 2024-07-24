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

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod vllm==0.5.3.post1

RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/Meta-Llama-3.1-8B-Instruct/raw/main/config.json -d /content/model -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/Meta-Llama-3.1-8B-Instruct/raw/main/generation_config.json -d /content/model -o generation_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/Meta-Llama-3.1-8B-Instruct/resolve/main/model-00001-of-00004.safetensors -d /content/model -o model-00001-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/Meta-Llama-3.1-8B-Instruct/resolve/main/model-00002-of-00004.safetensors -d /content/model -o model-00002-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/Meta-Llama-3.1-8B-Instruct/resolve/main/model-00003-of-00004.safetensors -d /content/model -o model-00003-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/Meta-Llama-3.1-8B-Instruct/resolve/main/model-00004-of-00004.safetensors -d /content/model -o model-00004-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/Meta-Llama-3.1-8B-Instruct/raw/main/model.safetensors.index.json -d /content/model -o model.safetensors.index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/Meta-Llama-3.1-8B-Instruct/raw/main/special_tokens_map.json -d /content/model -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/Meta-Llama-3.1-8B-Instruct/raw/main/tokenizer.json -d /content/model -o tokenizer.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/Meta-Llama-3.1-8B-Instruct/raw/main/tokenizer_config.json -d /content/model -o tokenizer_config.json
 
COPY ./worker_runpod.py /content/chat/worker_runpod.py
WORKDIR /content/chat
CMD python worker_runpod.py
