# ExllamaV2 Worker on Runpod Serverless

This is worker code which uses ExllamaV2 for inference on Runpod Serverless.

## 🌟 How to use
1. Clone this repository
1. build docker image
1. push docker image to your docker registry
1. deploy to Runpod Serverless

### 🏗️ build docker image
```bash
docker build -t <your docker registry>/<your docker image name>:<your docker image tag> .
```

These are the build arguments:

Note: The model gets downloaded on first run so these models get downloaded on runtime instead of on build. Make sure you are attaching network volume to serverless endpoint so download only happens once and the cache is used for subsequent runs

| key | value | optional |
| --- | --- | --- |
| HUGGING_FACE_HUB_TOKEN | your huggingface token | true |
| MODEL_NAME | your model name | false |
| MODEL_REVISION | your model revision | true |
| MODEL_BASE_PATH | your model base path | true |
| LORA_ADAPTER_NAME | your lora adapter name | true |
| LORA_ADAPTER_REVISION | your lora adapter revision | true |

### ⏫ push docker image to your docker registry
```bash
docker push <your docker registry>/<your docker image name>:<your docker image tag>
```

### 🚀 deploy to Runpod Serverless
After having docker image on your docker registry, you can deploy to Runpod Serverless.
