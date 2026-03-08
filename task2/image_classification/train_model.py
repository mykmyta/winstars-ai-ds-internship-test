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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def parse_args() -> argparse.Namespace:
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

    # Optional: cap dataset per class (useful for quick runs / balancing by subsampling)
    p.add_argument("--max-per-class", type=int, default=0, help="0 = no cap; otherwise limit images per class")
    return p.parse_args()


def extract_dataset(zip_path: Path, extract_dir: Path) -> None:
    """Extract zip only once. Assumes zip contains folder raw-img/ inside."""
    if extract_dir.exists():
        return
    extract_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir.parent)


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def merge_histories(*histories) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for h in histories:
        for k, v in h.history.items():
            merged.setdefault(k, []).extend(v)
    return merged


def save_training_plots(history: dict[str, list[float]], artifacts_dir: Path) -> None:
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
    """
    Creates train/val tf.data datasets. Uses Keras directory loader.
    Important: shuffle=True so that validation_split is not class-blocked.
    """
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
        shuffle=True,  # keeps the same split logic as training
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
    )

    class_names = train_ds.class_names

    # Optionally cap number of samples per class by filtering (simple, not perfect)
    # This is a helper for quick iteration; for "true" balancing use class_weight.
    if max_per_class and max_per_class > 0:
        train_ds = cap_per_class(train_ds, num_classes=len(class_names), max_per_class=max_per_class, seed=seed)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, class_names


def cap_per_class(ds: tf.data.Dataset, num_classes: int, max_per_class: int, seed: int) -> tf.data.Dataset:
    """
    Quick-and-dirty cap per class on an already batched dataset:
    - unbatch -> shuffle -> take per-class quota -> rebatch
    Not meant for huge datasets, but OK for quick experiments.
    """
    rng = tf.random.Generator.from_seed(seed)
    ds_unbatched = ds.unbatch()

    # shuffle a bit so we don't always take the same files
    ds_unbatched = ds_unbatched.shuffle(50_000, seed=seed, reshuffle_each_iteration=False)

    counts = tf.Variable(tf.zeros([num_classes], dtype=tf.int32), trainable=False)

    def _filter(x, y):
        yi = tf.cast(y, tf.int32)
        c = counts[yi]
        keep = c < max_per_class
        # update if keep
        def _inc():
            counts.scatter_nd_add(indices=[[yi]], updates=[1])
            return True
        def _no():
            return False
        return tf.cond(keep, _inc, _no)

    filtered = ds_unbatched.filter(_filter)
    return filtered.batch(ds.element_spec[1].shape[0] if hasattr(ds.element_spec[1], "shape") else 16).prefetch(tf.data.AUTOTUNE)


def compute_class_weight_from_ds(ds: tf.data.Dataset, num_classes: int, max_w: float = 10.0):
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, y in ds.unbatch():
        counts[int(y.numpy())] += 1

    total = counts.sum()
    raw = {i: float(total) / (num_classes * max(int(counts[i]), 1)) for i in range(num_classes)}
    class_weight = {i: min(w, max_w) for i, w in raw.items()}
    return class_weight, counts


def build_model(num_classes: int, img_size: tuple[int, int], lr_head: float):
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.1),   # helps with varying lighting
        ],
        name="data_augmentation",
    )

    # Try to request EfficientNet with built-in preprocessing (if supported)
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

    # AUTO: if base_model already has Rescaling/Normalization, do not add external Rescaling
    has_preprocessing = any(
        isinstance(l, (tf.keras.layers.Rescaling, tf.keras.layers.Normalization))
        for l in base_model.layers[:10]
    )

    inputs = tf.keras.Input(shape=img_size + (3,))
    x = data_augmentation(inputs)

    if not has_preprocessing:
        # if internal preprocessing is absent, add basic scaling
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
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
        ],
    )
    return model, base_model


def fine_tune(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    unfreeze_last: int,
    lr_finetune: float,
    label_smoothing: float,
):
    """
    Fine-tune last `unfreeze_last` layers of base model.
    IMPORTANT: keep BatchNorm layers frozen for stability.
    """
    base_model.trainable = True

    # Freeze BatchNorm layers always (stability)
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # Freeze all but last N layers
    if unfreeze_last > 0:
        for layer in base_model.layers[:-unfreeze_last]:
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_finetune),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
        ],
    )




def get_callbacks(artifacts_dir: Path):
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

    # Compute class weights (helps a LOT with imbalance)
    class_weight, counts = compute_class_weight_from_ds(train_ds, num_classes, max_w=10.0)
    print("\nTrain samples per class:")
    for name, c in zip(class_names, counts):
        print(f"{name:12s}: {c}")
    print("\nUsing class_weight:", {k: round(v, 4) for k, v in class_weight.items()})

    save_json(class_names, args.class_names_out)

    model, base_model = build_model(
        num_classes=num_classes,
        img_size=img_size,
        lr_head=args.lr_head,
    )

    callbacks = get_callbacks(args.artifacts_dir)

    # Stage 1: train head
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    # Stage 2: fine-tune
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

    # Save model + artifacts
    model.save(args.model_out)

    merged = merge_histories(history_head, history_ft)
    save_json(merged, args.artifacts_dir / "history.json")
    save_training_plots(merged, args.artifacts_dir)

    print(f"\nSaved model: {args.model_out}")
    print(f"Saved class names: {args.class_names_out}")
    print(f"Saved artifacts: {args.artifacts_dir}")


if __name__ == "__main__":
    main()


