from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate image classifier")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--class-names", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of the smallest class to use per class for balanced evaluation",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=0,
        help="Fixed number of samples per class for evaluation (0 = auto from --val-fraction)",
    )
    return parser.parse_args()


def load_class_names(path: Path) -> list[str]:
    """Load class names from a JSON artifact."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_class_images(class_dir: Path) -> list[Path]:
    """Collect image files from one class directory."""
    return [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def _choose_balanced_subset(
    data_dir: Path,
    class_names: list[str],
    seed: int,
    val_fraction: float,
    samples_per_class: int,
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    """Sample the same number of images from each class for balanced evaluation."""
    rng = np.random.default_rng(seed)
    files_by_class: dict[str, list[Path]] = {}

    for class_name in class_names:
        class_dir = data_dir / class_name
        if not class_dir.exists() or not class_dir.is_dir():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")
        files = _collect_class_images(class_dir)
        if not files:
            raise ValueError(f"No images found for class '{class_name}' in {class_dir}")
        files_by_class[class_name] = files

    min_class_count = min(len(files) for files in files_by_class.values())
    if samples_per_class > 0:
        n_per_class = min(samples_per_class, min_class_count)
    else:
        n_per_class = max(1, int(min_class_count * val_fraction))

    paths: list[str] = []
    labels: list[int] = []
    per_class_counts: dict[str, int] = {}

    for label_idx, class_name in enumerate(class_names):
        files = files_by_class[class_name]
        selected_idx = rng.choice(len(files), size=n_per_class, replace=False)
        selected = [files[i] for i in selected_idx]
        paths.extend(str(p) for p in selected)
        labels.extend([label_idx] * n_per_class)
        per_class_counts[class_name] = n_per_class

    order = rng.permutation(len(paths))
    paths = [paths[i] for i in order]
    labels = np.asarray([labels[i] for i in order], dtype=np.int32)

    return paths, labels, per_class_counts


def make_balanced_eval_dataset(
    data_dir: Path,
    img_size: tuple[int, int],
    batch_size: int,
    seed: int,
    class_names: list[str],
    val_fraction: float,
    samples_per_class: int,
):
    """Build a `tf.data.Dataset` from a balanced class subset."""
    paths, labels, per_class_counts = _choose_balanced_subset(
        data_dir=data_dir,
        class_names=class_names,
        seed=seed,
        val_fraction=val_fraction,
        samples_per_class=samples_per_class,
    )

    def _load_image(path: tf.Tensor, label: tf.Tensor):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, labels, per_class_counts


def main() -> None:
    """Evaluate a trained model and save confusion matrix artifact."""
    args = parse_args()
    img_size = (args.img_size, args.img_size)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(args.class_names)
    labels_all = list(range(len(class_names)))

    model = tf.keras.models.load_model(args.model)
    eval_ds, y_true, per_class_counts = make_balanced_eval_dataset(
        data_dir=args.data_dir,
        img_size=img_size,
        batch_size=args.batch_size,
        seed=args.seed,
        class_names=class_names,
        val_fraction=args.val_fraction,
        samples_per_class=args.samples_per_class,
    )

    eval_values = model.evaluate(eval_ds, verbose=1)
    if not isinstance(eval_values, (list, tuple)):
        eval_values = [eval_values]
    metrics = dict(zip(model.metrics_names, eval_values))

    print(f"\nValidation loss: {metrics.get('loss', float('nan')):.4f}")
    if "accuracy" in metrics:
        print(f"Validation accuracy: {metrics['accuracy']:.4f}")
    for name, value in metrics.items():
        if name not in {"loss", "accuracy"}:
            print(f"Validation {name}: {value:.4f}")

    probs = model.predict(eval_ds, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    print("\nBalanced eval samples per class:")
    for name in class_names:
        print(f"{name:12s}: {per_class_counts[name]}")

    print("\nClassification report (all classes):\n")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels_all,
            target_names=class_names,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels_all)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()

    print(f"\nSaved: {out_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
