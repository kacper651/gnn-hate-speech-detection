import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EvalPrediction
)
from sklearn.metrics import accuracy_score, f1_score

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallelism to avoid warnings
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Ensure CUDA operations are synchronous for debugging
# TORCH_CUDA_DSA
os.environ["TORCH_CUDA_DSA"] = "1"  # Enable CUDA DSA for better error handling

from sklearn.preprocessing import LabelEncoder
import numpy as np

encoder = LabelEncoder()
encoder.classes_ = np.load("../models/hateXplain/classes.npy", allow_pickle=True)

print(len(encoder.classes_))
print(encoder.classes_)

import json

with open("../models/hateXplain/dataset.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

from collections import Counter


def get_majority_label(annotators):
    labels = [a["label"] for a in annotators]
    most_common = Counter(labels).most_common(1)
    return most_common[0][0] if most_common else None


texts, labels = [], []

for post_id, post_data in raw_data.items():
    majority_label = get_majority_label(post_data["annotators"])
    if majority_label in encoder.classes_:
        label_id = encoder.transform([majority_label])[0]
        text = " ".join(post_data["post_tokens"])
        texts.append(text)
        labels.append(label_id)

print(f"Loaded {len(texts)} samples.")

# COnvert labels and texts to tensor
labels_tensor = torch.tensor(labels, dtype=torch.long)
print(f"Labels tensor shape: {labels_tensor.shape}")
print(f"Labels tensor dtype: {labels_tensor.dtype}")
print(f"Labels tensor: {labels_tensor[:10]}")
print(np.unique(labels_tensor, return_counts=True))

model_name = "vinai/bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)  # BERTweet uses Roberta tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels_tensor, test_size=0.2, random_state=42, stratify=labels_tensor
)

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

model_name = "vinai/bertweet-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)  # BERTweet uses Roberta tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 5. Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], truncation=True)

train_ds = train_dataset.map(tokenize, batched=True)
val_ds = val_dataset.map(tokenize, batched=True)

# 6. Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 7. Metrics
def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }

#Check for data mismatch
print(f"Train dataset size: {len(train_ds)}")
print(f"Validation dataset size: {len(val_ds)}")
print(train_ds[0])

# 8. Training arguments
training_args = TrainingArguments(
    output_dir="../models/hateXplain/reference/bertweet-3class",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    save_total_limit=2
)

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 10. Train
trainer.train()

# Optional: Save final model
trainer.save_model("bertweet-3class-finetuned")
