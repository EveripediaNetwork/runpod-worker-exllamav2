import torch
import logging, os, glob
from exllamav2.model import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config
from exllamav2.tokenizer import ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler
from schema import InferenceSettings
from download_model import download_model

MODEL_NAME = os.environ.get("MODEL_NAME")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")
MODEL_BASE_PATH = os.environ.get("MODEL_BASE_PATH", "/runpod-volume/")


class Predictor:
    def setup(self):
        # Model moved to network storage
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

        config = ExLlamaV2Config()
        config.model_dir = model_directory
        config.prepare()

        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading model, tokenizer and cache...")
        self.tokenizer = ExLlamaV2Tokenizer(config)
        self.model = ExLlamaV2(config)
        self.model.load()
        self.cache = ExLlamaV2Cache(self.model)
        self.generator = ExLlamaV2BaseGenerator(self.model, self.cache, self.tokenizer)

        self.settings = ExLlamaV2Sampler.Settings()
        self.inference_settings = InferenceSettings()

        self.settings.token_repetition_penalty_max = (
            self.inference_settings.token_repetition_penalty
        )
        self.settings.temperature = self.inference_settings.temperature
        self.settings.top_p = self.inference_settings.top_p
        self.settings.typical = self.inference_settings.typical_p
        self.settings.top_k = self.inference_settings.top_k
        self.settings.beams = self.inference_settings.num_beams
        self.settings.beam_length = self.inference_settings.length_penalty

    def predict(self, settings):
        return self.generate_to_eos(settings)

    def generate_to_eos(self, settings):
        print(settings)
        self.generator.warmup()
        max_new_tokens = 1000
        output = self.generator.generate_simple(
            settings["prompt"], self.settings, max_new_tokens, seed=1234
        )
        return output
