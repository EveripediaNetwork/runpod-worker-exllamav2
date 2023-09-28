import argparse
import sys
import runpod
from exllamav2 import (
    ExLlamaV2Cache,
    model_init,
)
from download_model import download_model
import torch
import os
import time

torch.cuda._lazy_init()
torch.set_printoptions(precision=10)

MODEL_NAME = os.environ.get("MODEL_NAME")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")
MODEL_BASE_PATH = os.environ.get("MODEL_BASE_PATH", "/runpod-volume/")

model_directory = f"{MODEL_BASE_PATH}{MODEL_NAME.split('/')[1]}"

# check if model directory exists. else, download model
if not os.path.isdir(model_directory):
    print("Downloading model...")
    try:
        download_model()
    except Exception as e:
        print(f"Error downloading model: {e}")
        # delete model directory if it exists
        if os.path.isdir(model_directory):
            os.system(f"rm -rf {model_directory}")
        raise e

torch.cuda._lazy_init()
torch.set_printoptions(precision=10)
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.set_float32_matmul_precision("medium")

parser = argparse.ArgumentParser(description="Test inference on ExLlamaV2 model")
# Initialize model and tokenizer

model_init.add_args(parser)
args = parser.parse_args()
args.model_dir = model_directory
model_init.print_options(args)
model, tokenizer = model_init.init(args)


# Test generation


def handler(event):
    prompt = event["input"]["prompt"]
    max_tokens = event["input"].get("max_tokens", 128)
    stop_tokens = event["input"].get("stop_token", ["</s>", tokenizer.decode(tokenizer.eos_token_id)])
    with torch.inference_mode():

        cache = ExLlamaV2Cache(model)

        ids = tokenizer.encode(prompt)
        tokens_prompt = ids.shape[-1]

        print(f" -- Warmup...")

        model.forward(ids[:, -1:])

        print(f" -- Generating (greedy sampling)...")
        print()
        print(prompt, end="")
        sys.stdout.flush()

        time_begin = time.time()

        if ids.shape[-1] > 1:
            model.forward(ids[:, :-1], cache, preprocess_only=True)

        torch.cuda.synchronize()
        time_prompt = time.time()

        output_list = []
        for i in range(max_tokens):
            text1 = tokenizer.decode(ids[:, -2:])[0]

            logits = model.forward(ids[:, -1:], cache)
            sample = torch.argmax(logits[0, -1]).cpu().unsqueeze(0).unsqueeze(0)
            ids = torch.cat((ids, sample), dim=-1)

            if tokenizer.decode(sample[0][0]) in stop_tokens:
                break

            output = tokenizer.decode(ids[:, -3:])[0]
            output = output[len(text1):]

            print(output, end="")
            print(i)
            output_list.append(output)  # append the generated output to the list

            # sys.stdout.flush()

        time_end = time.time()

    print()
    print()

    total_prompt = time_prompt - time_begin
    total_gen = time_end - time_prompt
    print(
        f"Prompt processed in {total_prompt:.2f} seconds, {tokens_prompt} tokens, {tokens_prompt / total_prompt:.2f} tokens/second")
    print(
        f"Response generated in {total_gen:.2f} seconds, {max_tokens} tokens, {max_tokens / total_gen:.2f} tokens/second")
    complete_output = "".join(output_list)
    return complete_output


runpod.serverless.start({"handler": handler})
