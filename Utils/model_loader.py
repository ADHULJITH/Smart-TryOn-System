"""
model_loader.py

Loads the Florence-2 model and processor.
"""

import torch

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM
)

from Utils.config import (
    FLORENCE_MODEL,
    DEVICE,
    DTYPE
)


def load_florence():
    """
    Load Florence-2 model and processor.
    """

    print("=" * 50)
    print("Loading Florence-2...")
    print("=" * 50)

    processor = AutoProcessor.from_pretrained(
        FLORENCE_MODEL,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        FLORENCE_MODEL,
        trust_remote_code=True,
        torch_dtype=DTYPE
    )

    model = model.to(DEVICE)

    model.eval()

    print("✓ Florence-2 loaded successfully.")
    print(f"Device : {DEVICE}")
    print(f"Model  : {FLORENCE_MODEL}")
    print("=" * 50)

    return processor, model