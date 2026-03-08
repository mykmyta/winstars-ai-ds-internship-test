from __future__ import annotations

import argparse
from pathlib import Path

from transformers import pipeline

from task2.utils.normalization import normalize_animal_name


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for NER inference."""
    parser = argparse.ArgumentParser(description="Run NER inference")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text", type=str, required=True)
    return parser.parse_args()


def extract_animal_from_text(text: str, model_dir: Path) -> dict:
    """Extract the first normalized animal mention from free text."""
    nlp = pipeline(
        "ner",
        model=str(model_dir),
        tokenizer=str(model_dir),
        aggregation_strategy="simple",
    )

    results = nlp(text)

    animal = None
    for item in results:
        entity_group = item.get("entity_group")
        word = item.get("word")
        if entity_group == "ANIMAL" or entity_group == "B-ANIMAL":
            animal = normalize_animal_name(word)
            break

    return {
        "text": text,
        "animal": animal,
        "raw_entities": results,
    }


def main() -> None:
    """CLI entrypoint for NER inference."""
    args = parse_args()
    result = extract_animal_from_text(args.text, args.model_dir)
    print(result)


if __name__ == "__main__":
    main()
