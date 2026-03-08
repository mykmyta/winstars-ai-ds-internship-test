# Task 2: Image Classification + NER + Claim Verification

This task contains three components:

1. `image_classification`: CNN-based animal image classifier (TensorFlow/Keras).
2. `ner`: token-classification model for extracting animal names from text (Transformers).
3. `pipeline`: combines image prediction + NER to verify whether text claim matches image.

## Project Structure

- `task2/image_classification/train_model.py` - train CNN model.
- `task2/image_classification/evaluate_model.py` - evaluate CNN and build confusion matrix.
- `task2/image_classification/infer_image.py` - run inference for one image.
- `task2/ner/train_ner.py` - train NER model from CoNLL-like dataset.
- `task2/ner/infer_ner.py` - run NER inference.
- `task2/pipeline/verify_claim.py` - end-to-end verification (text + image).
- `task2/demo.ipynb` - end-to-end demo notebook.

## Setup

```bash
cd task2
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Image classification expects:

- archive: `task2/image_classification/data/raw-img.zip`
- extracted folders by class under: `task2/image_classification/data/raw-img`

NER training expects:

- labeled file in CoNLL format (token + tag per line), e.g.:
  `task2/ner/data/conll_animal_dataset.txt`

## Train Image Classifier

Model training command used in this project:

```cmd
python -m task2.image_classification.train_model ^
  --zip-path task2/image_classification/data/raw-img.zip ^
  --extract-dir task2/image_classification/data/raw-img ^
  --model-out task2/image_classification/models/ANIMAL_CLF.keras ^
  --class-names-out task2/image_classification/artifacts/class_names.json ^
  --artifacts-dir task2/image_classification/artifacts ^
  --epochs-head 8 ^
  --epochs-finetune 20 ^
  --lr-head 1e-3 ^
  --lr-finetune 3e-5 ^
  --unfreeze-last 140 ^
  --batch-size 16
```

Expected artifacts:

- `task2/image_classification/models/ANIMAL_CLF.keras`
- `task2/image_classification/artifacts/class_names.json`
- `task2/image_classification/artifacts/history.json`
- `task2/image_classification/artifacts/accuracy.png`
- `task2/image_classification/artifacts/loss.png`

## Evaluate Image Classifier

```bash
python -m task2.image_classification.evaluate_model \
  --data-dir task2/image_classification/data/raw-img \
  --model task2/image_classification/models/ANIMAL_CLF.keras \
  --class-names task2/image_classification/artifacts/class_names.json \
  --out-dir task2/image_classification/artifacts \
  --samples-per-class 200
```

Expected output:

- console metrics (`loss`, `accuracy`, optional extra metrics)
- `task2/image_classification/artifacts/confusion_matrix.png`

## Single Image Inference

```bash
python -m task2.image_classification.infer_image \
  --model task2/image_classification/models/ANIMAL_CLF.keras \
  --class-names task2/image_classification/artifacts/class_names.json \
  --image task2/image_classification/data/raw-img/cat/OIP-...jpg
```

## Train NER

```bash
python -m task2.ner.train_ner \
  --data-path task2/ner/data/conll_animal_dataset.txt \
  --output-dir task2/ner/model \
  --base-model distilbert-base-uncased \
  --epochs 3 \
  --batch-size 8
```

## NER Inference

```bash
python -m task2.ner.infer_ner \
  --model-dir task2/ner/model \
  --text "This photo shows a squirrel on a tree."
```

## End-to-End Claim Verification

```bash
python -m task2.pipeline.verify_claim \
  --image task2/image_classification/data/raw-img/squirrel/OIP-...jpg \
  --text "This is a squirrel" \
  --cnn-model task2/image_classification/models/ANIMAL_CLF.keras \
  --class-names task2/image_classification/artifacts/class_names.json \
  --ner-model-dir task2/ner/model \
  --threshold 0.5
```

## Demo Notebook

Open and run `task2/demo.ipynb`.

Notebook includes:

- dataset overview and class distribution visualization
- training process visualization from saved `history.json`
- confusion matrix visualization
- standalone examples for image classifier and NER
- integration tests for `verify_claim` with matched and mismatched cases
