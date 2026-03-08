from __future__ import annotations

import argparse
from pathlib import Path

from task2.image_classification.infer_image import predict_image
from task2.ner.infer_ner import extract_animal_from_text
from task2.utils.normalization import normalize_animal_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify text claim against image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--cnn-model", type=Path, required=True)
    parser.add_argument("--class-names", type=Path, required=True)
    parser.add_argument("--ner-model-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--img-size", type=int, default=224)
    return parser.parse_args()


def verify_claim(
    image_path: Path,
    text: str,
    cnn_model: Path,
    class_names: Path,
    ner_model_dir: Path,
    threshold: float = 0.5,
    img_size: tuple[int, int] = (224, 224),
) -> dict:
    image_result = predict_image(
        image_path=image_path,
        model_path=cnn_model,
        class_names_path=class_names,
        img_size=img_size,
        top_k=3,
    )

    ner_result = extract_animal_from_text(text=text, model_dir=ner_model_dir)

    image_pred = normalize_animal_name(image_result["prediction"])
    image_conf = float(image_result["confidence"])
    text_animal = normalize_animal_name(ner_result["animal"])

    if text_animal is None:
        return {
            "match": False,
            "reason": "No animal found in text",
            "image_prediction": image_pred,
            "image_confidence": image_conf,
            "text_animal": None,
        }

    match = (image_pred == text_animal) and (image_conf >= threshold)

    return {
        "match": match,
        "reason": None,
        "image_prediction": image_pred,
        "image_confidence": image_conf,
        "text_animal": text_animal,
        "top_predictions": image_result["top_predictions"],
    }


def main() -> None:
    args = parse_args()
    result = verify_claim(
        image_path=args.image,
        text=args.text,
        cnn_model=args.cnn_model,
        class_names=args.class_names,
        ner_model_dir=args.ner_model_dir,
        threshold=args.threshold,
        img_size=(args.img_size, args.img_size),
    )
    print(result)


if __name__ == "__main__":
    main()