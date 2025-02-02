from datasets import load_from_disk
from PIL import Image
import numpy as np


def preprocess_image(image):
    """
    convert image to grayscale, resize and normalize
    """
    # covert to grayscale
    if isinstance(image, Image.Image):
        img = image.convert("L")
    else:
        img = Image.open(image).convert("L")
    
    # resize and normalize
    img = img.resize((28, 28))
    preprocessed_image = np.array(img) / 255.0

    return preprocessed_image


def prepare_split(dataset_split):
    """
    Split data for training, validation or testing.
    """
    images = [preprocess_image(sample["image"]) for sample in dataset_split]
    labels = [sample["label"] for sample in dataset_split]
    images = np.array(images).reshape(-1, 28 * 28)
    labels = np.array(labels)

    return images, labels
