# %% [markdown]
# # ISOM5240 Fine-Tuning Notebook
#
# This notebook fine-tunes a Hugging Face pre-trained model for financial news sentiment classification.
#
# Final labels:
# - Negative
# - Neutral
# - Positive

# %%
!pip -q install transformers datasets accelerate scikit-learn safetensors huggingface_hub sentencepiece tiktoken

# %%
import inspect
import json
import numpy as np
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
    set_seed,
)

# %%
BASE_MODEL = "distilbert-base-uncased"
DATASET_NAME = "atrost/financial_phrasebank"
DATASET_CONFIG = ""
OUTPUT_DIR = "finnews_distilbert_sentiment"
SEED = 42

LABEL_NAMES = ["Negative", "Neutral", "Positive"]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_NAMES)}
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_NAMES)}

set_seed(SEED)

# %% [markdown]
# ## Load Dataset
#
# The dataset contains financial sentences labelled as negative, neutral, or positive.

# %%
dataset_splits = load_dataset(DATASET_NAME)
dataset_splits

# %%
dataset_splits

# %%
for split_name, split_data in dataset_splits.items():
    print(split_name, len(split_data))

# %% [markdown]
# ## Tokenize Text

# %%
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)

def tokenize(batch):
    return tokenizer(batch["sentence"], truncation=True, max_length=128)

tokenized = dataset_splits.map(tokenize, batched=True)

# %% [markdown]
# ## Load Pre-Trained Model

# %%
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(LABEL_NAMES),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

# %%
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }

def make_training_args():
    kwargs = {
        "output_dir": OUTPUT_DIR,
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "num_train_epochs": 4,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "logging_steps": 25,
        "report_to": "none",
        "seed": SEED,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return TrainingArguments(**kwargs)

training_args = make_training_args()

# %% [markdown]
# ## Fine-Tune Model

# %%
trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": tokenized["train"],
    "eval_dataset": tokenized["validation"],
    "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
    "compute_metrics": compute_metrics,
}
trainer_signature = inspect.signature(Trainer.__init__)
if "tokenizer" in trainer_signature.parameters:
    trainer_kwargs["tokenizer"] = tokenizer
elif "processing_class" in trainer_signature.parameters:
    trainer_kwargs["processing_class"] = tokenizer

trainer = Trainer(**trainer_kwargs)

trainer.train()

# %% [markdown]
# ## Evaluate Final Model

# %%
validation_metrics = trainer.evaluate(tokenized["validation"])
test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")

print("Validation metrics")
print(validation_metrics)
print("Test metrics")
print(test_metrics)

# %% [markdown]
# ## Save Model Files

# %%
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

summary = {
    "base_model": BASE_MODEL,
    "dataset": DATASET_NAME,
    "labels": LABEL_NAMES,
    "num_train_samples": len(dataset_splits["train"]),
    "num_validation_samples": len(dataset_splits["validation"]),
    "num_test_samples": len(dataset_splits["test"]),
    "validation_metrics": validation_metrics,
    "test_metrics": test_metrics,
}

with open(f"{OUTPUT_DIR}/training_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

summary

# %% [markdown]
# ## Test with the Hugging Face Pipeline

# %%
sentiment_pipe = pipeline(
    "text-classification",
    model=OUTPUT_DIR,
    tokenizer=OUTPUT_DIR,
)

examples = [
    "Tesla shares fell after weaker-than-expected deliveries and rising competition in China.",
    "Apple reported stronger-than-expected revenue as services growth continued to support margins.",
    "The company announced that its board will meet next week to review the quarterly financial report.",
]

for text in examples:
    print(text)
    print(sentiment_pipe(text))
    print()

# %% [markdown]
# ## Upload to Hugging Face
#
# Run this section only after creating a Hugging Face write token.
# In Colab, use the left sidebar secret manager or paste the token when prompted.

# %%
from huggingface_hub import notebook_login

notebook_login()

# %%
# Replace this with your own Hugging Face username and model name.
HUB_MODEL_ID = "your-hf-username/finnews-distilbert-sentiment"

model.push_to_hub(HUB_MODEL_ID)
tokenizer.push_to_hub(HUB_MODEL_ID)

print(f"Model URL: https://huggingface.co/{HUB_MODEL_ID}")
