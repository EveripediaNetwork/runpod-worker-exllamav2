import os
from huggingface_hub import snapshot_download

# Get the hugging face token
HUGGING_FACE_HUB_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN", None)
MODEL_NAME = os.environ.get("MODEL_NAME")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")
MODEL_BASE_PATH = os.environ.get("MODEL_BASE_PATH", "/runpod-volume/")


def download_model():
    # Download the model from hugging face
    download_kwargs = {}

    if HUGGING_FACE_HUB_TOKEN:
        download_kwargs["token"] = HUGGING_FACE_HUB_TOKEN

    DOWNLOAD_PATH = f"{MODEL_BASE_PATH}{MODEL_NAME.split('/')[1]}"

    print(f"Downloading model to: {DOWNLOAD_PATH}")

    downloaded_path = snapshot_download(
        repo_id=MODEL_NAME,
        revision=MODEL_REVISION,
        local_dir=DOWNLOAD_PATH,
        local_dir_use_symlinks=False,
        **download_kwargs,
    )

    print(f"Finished downloading to: {downloaded_path}")
