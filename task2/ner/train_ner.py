import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import Dataset
from datasets import DatasetDict
from sklearn.model_selection import train_test_split

import os


data_path = "task2\\ner\\data\\conll_animal_dataset.txt"

with open(data_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

def process_conll_data(lines):
    sentences = []
    labels = []
    sentence = []
    label = []

    for line in lines:
        line = line.strip()
        if not line:  # An empty line indicates a new sentence
            if sentence:
                sentences.append(sentence)
                labels.append(label)
                sentence = []
                label = []
        else:
            parts = line.split()
            if len(parts) == 2:  # Ensure the line contains a word and a label
                word, tag = parts
                sentence.append(word)
                label.append(tag)
    
    # Add the last sentence if it was not added
    if sentence:
        sentences.append(sentence)
        labels.append(label)

    return sentences, labels

sentences, labels = process_conll_data(lines)



# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenization and label alignment
def tokenize_and_align_labels(sentences, labels, tokenizer):
    tokenized_inputs = tokenizer(
        sentences, 
        is_split_into_words=True, 
        padding="max_length", 
        truncation=True, 
        max_length=128, 
        return_tensors="pt"
    )

    word_ids = [tokenized_inputs.word_ids(batch_index=i) for i in range(len(sentences))]
    
    aligned_labels = []
    for i, word_id in enumerate(word_ids):
        previous_word_id = None
        label_ids = []
        for word_idx in word_id:
            if word_idx is None:
                label_ids.append(-100)  # Ignore special tokens
            elif word_idx != previous_word_id:
                label_ids.append(label2id[labels[i][word_idx]])
            else:
                label_ids.append(label2id[labels[i][word_idx]])  # Use the same label
            previous_word_id = word_idx
        aligned_labels.append(label_ids)

    return tokenized_inputs, aligned_labels

# Create mapping of labels to numeric IDs
unique_labels = set(tag for sublist in labels for tag in sublist)
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

# Perform tokenization and align labels
tokenized_inputs, aligned_labels = tokenize_and_align_labels(sentences, labels, tokenizer)


# Convert to Dataset
dataset = Dataset.from_dict({
    "input_ids": tokenized_inputs["input_ids"],
    "attention_mask": tokenized_inputs["attention_mask"],
    "labels": torch.tensor(aligned_labels)
})


train_sentences, val_sentences, train_labels, val_labels = train_test_split(sentences, labels, test_size=0.2, random_state=42)

train_tokenized, train_aligned_labels = tokenize_and_align_labels(train_sentences, train_labels, tokenizer)
val_tokenized, val_aligned_labels = tokenize_and_align_labels(val_sentences, val_labels, tokenizer)

# Convert to Dataset
train_dataset = Dataset.from_dict({
    "input_ids": train_tokenized["input_ids"],
    "attention_mask": train_tokenized["attention_mask"],
    "labels": torch.tensor(train_aligned_labels)
})

val_dataset = Dataset.from_dict({
    "input_ids": val_tokenized["input_ids"],
    "attention_mask": val_tokenized["attention_mask"],
    "labels": torch.tensor(val_aligned_labels)
})

# Combine into DatasetDict
dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})



# Load model
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Training configuration
training_args = TrainingArguments(
    output_dir="./ner_animal_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True
)

# Collator for handling sentence lengths
data_collator = DataCollatorForTokenClassification(tokenizer)

# Configure Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start training
trainer.train()

# Save model and tokenizer
trainer.save_model("task2\\ner\\ner-animal-model")  # Saves the model (equivalent to model.save_pretrained)
tokenizer.save_pretrained("task2\\ner\\ner-animal-tokenizer")
