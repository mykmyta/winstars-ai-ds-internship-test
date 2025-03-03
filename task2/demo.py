import inflect
import tensorflow as tf
from transformers import pipeline
from tensorflow.keras import models #type: ignore # Load CNN model
from tensorflow.keras.preprocessing import image #type: ignore # Image processing
import numpy as np

# Load models
cnn_model = models.load_model("task2/image_classsification/ANIMAL_CLF.keras")
nlp = pipeline("ner", model="task2/ner/ner-animal-model", tokenizer="task2/ner/ner-animal-tokenizer", aggregation_strategy="simple")

def load_class_labels(file_path):
    """Load class labels for CNN"""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f.readlines()]

# Load class labels
class_labels = load_class_labels("task2/image_classsification/data/raw-img/name of the animals.txt")

def extract_animal_from_text(text):
    """Extract animal names from text using NER"""
    results = nlp(text)
    animals = []
    current_animal = ""

    for res in results:
        token = res['word']
        if token.startswith("##"):
            current_animal += token[2:]  # Append without "##"
        else:
            if current_animal:  
                animals.append(current_animal)  # Add previous word
            current_animal = token  # Start new word
    
    if current_animal:
        animals.append(current_animal)

    animals = [animal.lower() for animal in animals]

    return animals if animals else None

def singularize(words):
    """Convert words to singular form"""
    p = inflect.engine()
    singular_word = [p.singular_noun(word) if p.singular_noun(word) else word for word in words]
    return singular_word

def classify_image(img_path):
    """Classify image using CNN"""
    try:
        img = image.load_img('/'.join(["task2/photos/", img_path]), target_size=(224, 224))
    except:
        print("Error loading image.")
        return None
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = cnn_model.predict(img_array, verbose=0)
    predicted_label = class_labels[np.argmax(predictions)]
    return predicted_label

def verify_animal(text, image_path):
    """Verify correspondence between text and image"""
    extracted_animal = extract_animal_from_text(text)
    if not extracted_animal:
        print("No animal found in text.")
        return False
    
    extracted_animal = singularize(extracted_animal)
    predicted_animal = classify_image(image_path)
    
    print(f"Extracted from text: {extracted_animal}")
    print(f"Image classification: {predicted_animal}")
    
    return predicted_animal in extracted_animal

# Example usage
while True:
    text_input = input("Enter text: ")
    image_file = input("Enter image file name: ")
    
    result = verify_animal(text_input, image_file)
    print("Verification result:", result)
    print()