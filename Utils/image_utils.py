"""
image_utils.py
"""

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def load_image(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found:\n{path}")

    return Image.open(path).convert("RGB")


def save_image(image, path):

    image.save(path)


def resize_image(image, width, height):

    return image.resize((width, height))


def show_image(image, title="Image"):

    plt.figure(figsize=(5,7))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()