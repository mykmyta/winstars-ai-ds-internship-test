from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image inference")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--class-names", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def load_class_names(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_image(image_path: Path, img_size: tuple[int, int]) -> np.ndarray:
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_image(
    image_path: Path,
    model_path: Path,
    class_names_path: Path,
    img_size: tuple[int, int] = (224, 224),
    top_k: int = 3,
) -> dict:
    model = tf.keras.models.load_model(model_path)
    class_names = load_class_names(class_names_path)

    x = preprocess_image(image_path, img_size)
    probs = model.predict(x, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    top_indices = np.argsort(probs)[::-1][:top_k]
    top_predictions = [
        {"label": class_names[int(i)], "score": float(probs[int(i)])}
        for i in top_indices
    ]

    return {
        "image_path": str(image_path),
        "prediction": pred_label,
        "confidence": confidence,
        "top_predictions": top_predictions,
    }


def main() -> None:
    args = parse_args()
    result = predict_image(
        image_path=args.image,
        model_path=args.model,
        class_names_path=args.class_names,
        img_size=(args.img_size, args.img_size),
        top_k=args.top_k,
    )
    print(result)


if __name__ == "__main__":
    main()