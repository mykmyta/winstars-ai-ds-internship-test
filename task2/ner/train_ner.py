from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


LABEL_LIST = ["O", "B-ANIMAL"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NER model")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def read_conll(path: Path):
    sentences = []
    labels = []

    current_tokens = []
    current_tags = []

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                if current_tokens:
                    sentences.append(current_tokens)
                    labels.append(current_tags)
                    current_tokens = []
                    current_tags = []
                continue

            token, tag = line.split()
            current_tokens.append(token)
            current_tags.append(LABEL2ID[tag])

    if current_tokens:
        sentences.append(current_tokens)
        labels.append(current_tags)

    return sentences, labels


def build_dataset(sentences, labels) -> DatasetDict:
    dataset = Dataset.from_dict({"tokens": sentences, "ner_tags": labels})
    split = dataset.train_test_split(test_size=0.2, seed=42)
    return DatasetDict({"train": split["train"], "test": split["test"]})


def tokenize_and_align_labels(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )

    aligned_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(labels[word_id])
            else:
                label_ids.append(-100)
            prev_word_id = word_id

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


def main() -> None:
    args = parse_args()

    sentences, labels = read_conll(args.data_path)
    dataset = build_dataset(sentences, labels)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "training_runs"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=str(args.output_dir / "logs"),
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,   # modern replacement for tokenizer=
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    trainer.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Saved NER model to: {args.output_dir}")


if __name__ == "__main__":
    main()
