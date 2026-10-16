"""Predict the waste category of one image using a saved Keras model.

Example (PowerShell):
  python predict.py --image sample_images/my_photo.jpg
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "models" / "best_waste_classifier.keras"
DEFAULT_MAPPING = PROJECT_DIR / "models" / "class_mapping.json"
IMAGE_SIZE = (160, 160)


def predict_image(image_path: Path, model_path: Path, mapping_path: Path) -> dict:
    """Load a trained model and predict one image."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train the notebook first or copy a trained model here."
        )
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Class mapping not found: {mapping_path}")

    model = keras.models.load_model(model_path)
    with mapping_path.open("r", encoding="utf-8") as file:
        class_to_index = json.load(file)
    index_to_class = {int(index): name for name, index in class_to_index.items()}

    image_bytes = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
    image = tf.cast(image, tf.float32) / 255.0

    probabilities = model.predict(tf.expand_dims(image, axis=0), verbose=0)[0]
    best_index = int(np.argmax(probabilities))
    return {
        "predicted_class": index_to_class[best_index],
        "confidence": float(probabilities[best_index]),
    }


def main():
    parser = argparse.ArgumentParser(description="Classify a waste image into one of six TrashNet classes.")
    parser.add_argument("--image", required=True, type=Path, help="Path to the image to classify")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to a .keras model")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING, help="Path to class_mapping.json")
    args = parser.parse_args()

    result = predict_image(args.image, args.model, args.mapping)
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    main()
