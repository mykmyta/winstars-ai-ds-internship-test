# task2/image_classification/train_model.py
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for image-classification training."""
    p = argparse.ArgumentParser(description="Train animal image classifier (EfficientNetB0)")
    p.add_argument("--zip-path", type=Path, required=True)
    p.add_argument("--extract-dir", type=Path, required=True)
    p.add_argument("--model-out", type=Path, required=True)
    p.add_argument("--class-names-out", type=Path, required=True)
    p.add_argument("--artifacts-dir", type=Path, required=True)

    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epochs-head", type=int, default=10)
    p.add_argument("--epochs-finetune", type=int, default=15)

    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-finetune", type=float, default=1e-5)

    p.add_argument("--unfreeze-last", type=int, default=80, help="How many last base layers to unfreeze")
    p.add_argument("--label-smoothing", type=float, default=0.05)

    p.add_argument("--max-per-class", type=int, default=0, help="0 = no cap; otherwise limit images per class")
    return p.parse_args()


def extract_dataset(zip_path: Path, extract_dir: Path) -> None:
    """Extract the dataset archive once if the target folder does not yet exist."""
    if extract_dir.exists():
        return
    extract_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir.parent)


def save_json(data, path: Path) -> None:
    """Serialize a Python object to UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def merge_histories(*histories) -> dict[str, list[float]]:
    """Merge multiple Keras `History` objects into a single metrics dictionary."""
    merged: dict[str, list[float]] = {}
    for history in histories:
        for metric_name, values in history.history.items():
            merged.setdefault(metric_name, []).extend(values)
    return merged


def save_training_plots(history: dict[str, list[float]], artifacts_dir: Path) -> None:
    """Save accuracy and loss curves from merged training history."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if "accuracy" in history or "val_accuracy" in history:
        plt.figure(figsize=(8, 5))
        if "accuracy" in history:
            plt.plot(history["accuracy"], label="train_accuracy")
        if "val_accuracy" in history:
            plt.plot(history["val_accuracy"], label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(artifacts_dir / "accuracy.png")
        plt.close()

    if "loss" in history or "val_loss" in history:
        plt.figure(figsize=(8, 5))
        if "loss" in history:
            plt.plot(history["loss"], label="train_loss")
        if "val_loss" in history:
            plt.plot(history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(artifacts_dir / "loss.png")
        plt.close()


def make_datasets(
    extract_dir: Path,
    img_size: tuple[int, int],
    batch_size: int,
    seed: int,
    max_per_class: int = 0,
):
    """Create train and validation datasets from class folders."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        extract_dir,
        validation_split=0.2,
        subset="training",
        seed=seed,
        shuffle=True,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        extract_dir,
        validation_split=0.2,
        subset="validation",
        seed=seed,
        shuffle=True,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
    )

    class_names = train_ds.class_names

    if max_per_class and max_per_class > 0:
        train_ds = cap_per_class(
            train_ds,
            num_classes=len(class_names),
            max_per_class=max_per_class,
            seed=seed,
            batch_size=batch_size,
        )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names


def cap_per_class(
    ds: tf.data.Dataset,
    num_classes: int,
    max_per_class: int,
    seed: int,
    batch_size: int,
) -> tf.data.Dataset:
    """Limit the training dataset to at most `max_per_class` samples per class."""
    ds_unbatched = ds.unbatch().shuffle(50_000, seed=seed, reshuffle_each_iteration=False)
    counts = tf.Variable(tf.zeros([num_classes], dtype=tf.int32), trainable=False)

    def _filter(_, y):
        yi = tf.cast(y, tf.int32)
        current = counts[yi]
        keep = current < max_per_class

        def _inc():
            counts.scatter_nd_add(indices=[[yi]], updates=[1])
            return True

        def _no():
            return False

        return tf.cond(keep, _inc, _no)

    filtered = ds_unbatched.filter(_filter)
    return filtered.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def compute_class_weight_from_ds(ds: tf.data.Dataset, num_classes: int, max_w: float = 10.0):
    """Compute inverse-frequency class weights with an upper cap."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, y in ds.unbatch():
        counts[int(y.numpy())] += 1

    total = counts.sum()
    raw = {i: float(total) / (num_classes * max(int(counts[i]), 1)) for i in range(num_classes)}
    class_weight = {i: min(weight, max_w) for i, weight in raw.items()}
    return class_weight, counts


def build_model(num_classes: int, img_size: tuple[int, int], lr_head: float):
    """Build and compile a transfer-learning model based on EfficientNetB0."""
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )

    try:
        base_model = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=img_size + (3,),
            include_preprocessing=True,
        )
    except TypeError:
        base_model = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=img_size + (3,),
        )

    base_model.trainable = False

    has_preprocessing = any(
        isinstance(layer, (tf.keras.layers.Rescaling, tf.keras.layers.Normalization))
        for layer in base_model.layers[:10]
    )

    inputs = tf.keras.Input(shape=img_size + (3,))
    x = data_augmentation(inputs)

    if not has_preprocessing:
        x = layers.Rescaling(1.0 / 255.0)(x)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_head),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3")],
    )
    return model, base_model


def fine_tune(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    unfreeze_last: int,
    lr_finetune: float,
    label_smoothing: float,
) -> None:
    """Unfreeze the tail of the backbone and recompile the model for fine-tuning."""
    base_model.trainable = True

    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    if unfreeze_last > 0:
        for layer in base_model.layers[:-unfreeze_last]:
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_finetune),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3")],
    )


def get_callbacks(artifacts_dir: Path):
    """Create a standard callback list for checkpointing and early stopping."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return [
        ModelCheckpoint(
            filepath=str(artifacts_dir / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
    ]


def main() -> None:
    """Run full training workflow and save model + artifacts."""
    args = parse_args()

    img_size = (args.img_size, args.img_size)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.class_names_out.parent.mkdir(parents=True, exist_ok=True)

    extract_dataset(args.zip_path, args.extract_dir)
    train_ds, val_ds, class_names = make_datasets(
        extract_dir=args.extract_dir,
        img_size=img_size,
        batch_size=args.batch_size,
        seed=args.seed,
        max_per_class=args.max_per_class,
    )

    num_classes = len(class_names)
    print("Classes:", class_names)

    class_weight, counts = compute_class_weight_from_ds(train_ds, num_classes, max_w=10.0)
    print("\nTrain samples per class:")
    for name, count in zip(class_names, counts):
        print(f"{name:12s}: {count}")
    print("\nUsing class_weight:", {k: round(v, 4) for k, v in class_weight.items()})

    save_json(class_names, args.class_names_out)

    model, base_model = build_model(
        num_classes=num_classes,
        img_size=img_size,
        lr_head=args.lr_head,
    )

    callbacks = get_callbacks(args.artifacts_dir)

    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    fine_tune(
        model=model,
        base_model=base_model,
        unfreeze_last=args.unfreeze_last,
        lr_finetune=args.lr_finetune,
        label_smoothing=args.label_smoothing,
    )

    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    model.save(args.model_out)

    merged = merge_histories(history_head, history_ft)
    save_json(merged, args.artifacts_dir / "history.json")
    save_training_plots(merged, args.artifacts_dir)

    print(f"\nSaved model: {args.model_out}")
    print(f"Saved class names: {args.class_names_out}")
    print(f"Saved artifacts: {args.artifacts_dir}")


if __name__ == "__main__":
    main()
