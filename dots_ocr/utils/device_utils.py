import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_attn_implementation():
    if torch.cuda.is_available():
        return "flash_attention_2"
    else:
        # Mac and CPU don't support flash_attention_2
        return "sdpa"
