import zipfile
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetB3  #type: ignore # Import pre-trained models
from tensorflow.keras import models, layers #type: ignore  # For building neural networks
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore  # Data augmentation
from tensorflow.keras.callbacks import ReduceLROnPlateau #type: ignore  # Learning rate scheduler

# Define the path to the dataset
zip_path = "data\\raw-img.zip"

# Extract image file names from the archive
with zipfile.ZipFile(zip_path, 'r') as archive:
    file_list = [f for f in archive.namelist() if f.endswith(('jpg', 'jpeg', 'png'))]

# Extract unique class labels
labels = set()
for file in file_list:
    labels.add(file.split("/")[2])  # Assuming class labels are in the second folder level
id2label = {i: label for i, label in enumerate(sorted(list(labels)))}
label2id = {label: i for i, label in id2label.items()}

# Initialize lists for images and labels
image_data = []
labels = []

# Read and preprocess images from the archive
with zipfile.ZipFile(zip_path, 'r') as archive:
    for file in file_list:
        file_content = archive.read(file)
        image = tf.io.decode_image(file_content, channels=3)  # Ensure 3-channel images
        image = tf.image.resize(image, (224, 224))  # Resize to match model input
        image = image / 255.0  # Normalize pixel values
        image_data.append(image)
        labels.append(label2id[file.split('/')[2]])  # Assign numerical labels

# Convert lists to NumPy arrays
image_data = np.array(image_data)
labels = np.array(labels)

# Shuffle data
indices = np.arange(len(labels))
np.random.shuffle(indices)
image_data = image_data[indices]
labels = labels[indices]

# Split into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(image_data, labels, train_size=0.8, random_state=42)

# Data augmentation for training images
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load the EfficientNetB3 model with pre-trained weights
base_model1 = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze some of the top layers of EfficientNetB3 for fine-tuning
for layer in base_model1.layers[-50:]:  # Fine-tune last 50 layers
    layer.trainable = True

# Build the classification model
model1 = models.Sequential([
    base_model1,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(90, activation='softmax')  # Output layer for 90 classes
])

# Compile the model with a lower learning rate for fine-tuning
model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Learning rate scheduler to reduce learning rate on plateaus
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# Train the model with data augmentation and learning rate scheduling
history = model1.fit(datagen.flow(train_images, train_labels, batch_size=32), 
                    epochs=10, 
                    validation_data=(train_images, train_labels), 
                    callbacks=[lr_scheduler])

# Evaluate model performance
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = np.argmax(model1.predict(test_images), axis=1)

print("Accuracy:", accuracy_score(test_labels, y_pred))

# Save the trained model
model1.save("ANIMAL_CLF.keras")
