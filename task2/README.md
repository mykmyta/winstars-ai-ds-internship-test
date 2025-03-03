# Animal Image Classification and Named Entity Recognition (NER)

This project involves two main tasks:
1. Classifying images of animals using a Convolutional Neural Network (CNN).
2. Extracting animal names from text using a Named Entity Recognition (NER) model.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-repo/winstars-ai-ds-internship-test.git
    cd winstars-ai-ds-internship-test/task2
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Download the necessary data and models:
    - Place the raw image dataset (`raw-img.zip`) in the `data` directory.
    - Ensure the NER model and tokenizer are available in the `ner` directory.

## Usage

### Training the NER Model

To train the NER model, run:
```sh
python ner/train_ner.py
```

### Training the Image Classification Model

To train the image classification model, run:
```sh
python image_classsification/img_clf.py
```

### Running the Demo

To run the demo that verifies the correspondence between text and image, run:
```sh
python demo.py
```

You will be prompted to enter a text description and the name of an image file. The script will then verify if the animal mentioned in the text matches the animal in the image.

### Example

```sh
Enter text: The picture shows a cow.
Enter image file name: cow.jpg
Extracted from text: ['cow']
Image classification: cow
Verification result: True
```

## Data

- The `data/raw-img.zip` file contains images of various animals.
- The `ner/data/conll_animal_dataset.txt` file contains the dataset for training the NER model.

## Models

- The image classification model is saved as `ANIMAL_CLF.keras`.
- The NER model and tokenizer are saved in the `ner` directory.

## License

This project is licensed under the MIT License.
