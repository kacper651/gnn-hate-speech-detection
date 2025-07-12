import pandas as pd
import torch
from sklearn.model_selection import RepeatedStratifiedKFold
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import json
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = LabelEncoder()
encoder.classes_ = np.load("../models/hateXplain/classes.npy", allow_pickle=True)

with open("../models/hateXplain/dataset.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)


def get_majority_label(annotators):
    labels = [a["label"] for a in annotators]
    most_common = Counter(labels).most_common(1)
    return most_common[0][0] if most_common else None


texts, labels = [], []
for post_id, post_data in raw_data.items():
    majority_label = get_majority_label(post_data["annotators"])
    if majority_label in encoder.classes_:
        text = " ".join(post_data["post_tokens"])
        texts.append(text)
        labels.append(int(encoder.transform([majority_label])[0]))

pd.DataFrame({"text": texts, "label": labels}).to_csv(
    "../output/csv/reference/hateXplain_dataset.csv", index=False
)

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

model_infos = [
    ("bertweet", "vinai/bertweet-base"),
    ("allminilm", "sentence-transformers/all-MiniLM-L6-v2"),
]

results = []

for model_key, model_name in model_infos:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for fold, (train_idx, val_idx) in enumerate(rskf.split(texts, labels)):
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]

        train_enc = (
            tokenizer(
                train_texts, truncation=True, padding="max_length", max_length=128
            )
            if model_key == "bertweet"
            else tokenizer(train_texts, truncation=True, padding=True)
        )
        val_enc = (
            tokenizer(val_texts, truncation=True, padding="max_length", max_length=128)
            if model_key == "bertweet"
            else tokenizer(val_texts, truncation=True, padding=True)
        )

        class HSDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        train_ds = HSDataset(train_enc, train_labels)
        val_ds = HSDataset(val_enc, val_labels)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(encoder.classes_)
        ).to(device)

        args = TrainingArguments(
            output_dir=f"./results/{model_key}/fold_{fold}",
            num_train_epochs=100,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
        )

        def compute_metrics(p):
            probs = softmax(p.predictions, axis=1)
            preds = np.argmax(probs, axis=1)
            return {
                "accuracy": accuracy_score(p.label_ids, preds),
                "f1": f1_score(p.label_ids, preds, average="macro"),
                "eval_roc_auc": roc_auc_score(
                    p.label_ids, probs, multi_class="ovr", average="macro"
                ),
            }

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(
                tokenizer, pad_to_multiple_of=8, padding=True
            ),
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=20, early_stopping_threshold=0.01
                )
            ],
        )

        # Train and evaluate
        trainer.train()
        metrics = trainer.evaluate()

        # Extract logs
        log = trainer.state.log_history
        train_losses = [x["loss"] for x in log if "loss" in x]
        val_losses = [x["eval_loss"] for x in log if "eval_loss" in x]
        val_accuracies = [x["eval_accuracy"] for x in log if "eval_accuracy" in x]
        val_f1s = [x["eval_f1"] for x in log if "eval_f1" in x]
        val_roc_aucs = [x["eval_roc_auc"] for x in log if "eval_roc_auc" in x]

        results.append(
            {
                "model": model_key,
                "fold": fold + 1,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_accuracies": val_accuracies,
                "val_f1s": val_f1s,
                "val_roc_aucs": val_roc_aucs,
                "best_val_loss": min(val_losses) if val_losses else None,
                "test_loss": metrics["eval_loss"],
                "test_acc": metrics["eval_accuracy"],
                "test_f1": metrics["eval_f1"],
                "test_roc_auc": metrics["eval_roc_auc"],
            }
        )

pd.DataFrame(results).to_csv(
    "../output/csv/reference/reference_results_two_models.csv", index=False
)
