# Base image
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

# Use bash shell with pipefail option
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set the working directory

RUN mkdir data
WORKDIR /data


# Prepare the models inside the docker image
ARG HUGGING_FACE_HUB_TOKEN=
ENV HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN

# Prepare argument for the model and tokenizer
ARG MODEL_NAME=""
ENV MODEL_NAME=$MODEL_NAME
ARG MODEL_REVISION="main"
ENV MODEL_REVISION=$MODEL_REVISION
ARG MODEL_BASE_PATH="/runpod-volume/"
ENV MODEL_BASE_PATH=$MODEL_BASE_PATH
ENV HF_HOME="/runpod-volume"

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r requirements.txt --no-cache-dir && \
    rm requirements.txt

# Add src files (Worker Template)
RUN git clone https://github.com/turboderp/exllamav2
RUN pip install -r exllamav2/requirements.txt

COPY __init.py__ /data/__init__.py
COPY ./src/* /data/

ENV PYTHONPATH=/data/exllamav2

CMD [ "python", "-m", "runpod_handler"]
