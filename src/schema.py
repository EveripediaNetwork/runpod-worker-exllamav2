INPUT_SCHEMA = {
    "prompt": {
        "type": str,
        "required": True,
    },
    "temperature": {"type": float, "required": False, "default": 1.31},
    "top_p": {"type": float, "required": False, "default": 0.14},
    "top_k": {"type": int, "required": False, "default": 49},
    "max_new_tokens": {"type": int, "required": False, "default": 1024},
    "token_repetition_penalty": {"type": float, "required": False, "default": 1.15},
    "token_repetition_range": {"type": int, "required": False, "default": -1},
    "token_repetition_decay": {"type": int, "required": False, "default": 0},
}
