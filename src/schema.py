class InferenceSettings:
    def __init__(self, **kwargs):
        # Set the default values for all settings
        self.prompt = "output EOS token"
        self.reverse_prompt = ["###"]
        self.temperature = 1.31
        self.top_p = 0.14
        self.top_k = 49
        self.typical_p = 1.0
        self.max_new_tokens = 1024
        self.token_repetition_penalty = 1.17
        self.tail_free_sampling = 1.0
        self.num_beams = 1
        self.length_penalty = 1.0

        # Overwrite the default values with any provided in kwargs
        for key, value in kwargs.items():
            # Check if the key is a valid setting name
            if hasattr(self, key):
                # Check if the value matches the expected type
                if isinstance(value, type(getattr(self, key))):
                    # Set the attribute to the value
                    setattr(self, key, value)
                else:
                    # Raise an exception if the type is wrong
                    raise TypeError(
                        f"Expected {type(getattr(self, key))} for {key}, got {type(value)}"
                    )
            else:
                # Raise an exception if the key is invalid
                raise AttributeError(f"Invalid setting name: {key}")


INPUT_SCHEMA = {
    "prompt": {
        "type": str,
        "required": True,
    },
    "reverse_prompt": {"type": list, "required": False, "default": ["###"]},
    "temperature": {"type": float, "required": False, "default": 1.31},
    "top_p": {"type": float, "required": False, "default": 0.14},
    "top_k": {"type": int, "required": False, "default": 49},
    "typical_p": {"type": float, "required": False, "default": 1.0},
    "max_new_tokens": {"type": int, "required": False, "default": 1024},
    "token_repetition_penalty": {"type": float, "required": False, "default": 1.17},
    "tail_free_sampling": {"type": float, "required": False, "default": 1.0},
    "num_beams": {"type": int, "required": False, "default": 1},
    "length_penalty": {"type": float, "required": False, "default": 1.0},
}
