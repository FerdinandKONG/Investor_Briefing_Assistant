import argparse
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
    set_seed,
)


LABEL_NAMES = ["Negative", "Neutral", "Positive"]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_NAMES)}
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_NAMES)}


def normalize_label(value):
    text = str(value).strip().lower()
    mapping = {
        "0": 0,
        "negative": 0,
        "neg": 0,
        "1": 1,
        "neutral": 1,
        "neu": 1,
        "2": 2,
        "positive": 2,
        "pos": 2,
    }
    if text not in mapping:
        raise ValueError(f"Unsupported label value: {value!r}")
    return mapping[text]


def load_project_dataset(args):
    if args.train_csv:
        df = pd.read_csv(args.train_csv)
        if args.text_column not in df.columns:
            raise ValueError(f"Missing text column: {args.text_column}")
        if args.label_column not in df.columns:
            raise ValueError(f"Missing label column: {args.label_column}")

        df = df[[args.text_column, args.label_column]].dropna()
        df = df.rename(columns={args.text_column: "sentence", args.label_column: "label"})
        df["label"] = df["label"].map(normalize_label)
        dataset = Dataset.from_pandas(df, preserve_index=False)
        if args.sample_limit:
            dataset = dataset.shuffle(seed=args.seed).select(range(min(args.sample_limit, len(dataset))))
    else:
        if args.dataset_config:
            loaded_dataset = load_dataset(args.dataset_name, args.dataset_config)
        else:
            loaded_dataset = load_dataset(args.dataset_name)

        if isinstance(loaded_dataset, DatasetDict) and {
            "train",
            "validation",
            "test",
        }.issubset(loaded_dataset.keys()):
            if args.sample_limit:
                validation_limit = max(1, min(len(loaded_dataset["validation"]), args.sample_limit // 4))
                test_limit = max(1, min(len(loaded_dataset["test"]), args.sample_limit // 4))
                return DatasetDict(
                    {
                        "train": loaded_dataset["train"]
                        .shuffle(seed=args.seed)
                        .select(range(min(args.sample_limit, len(loaded_dataset["train"])))),
                        "validation": loaded_dataset["validation"]
                        .shuffle(seed=args.seed)
                        .select(range(validation_limit)),
                        "test": loaded_dataset["test"]
                        .shuffle(seed=args.seed)
                        .select(range(test_limit)),
                    }
                )
            return loaded_dataset

        dataset = loaded_dataset["train"] if isinstance(loaded_dataset, DatasetDict) else loaded_dataset

        if "sentence" not in dataset.column_names:
            raise ValueError("Expected a 'sentence' column in the Hugging Face dataset.")
        if "label" not in dataset.column_names:
            raise ValueError("Expected a 'label' column in the Hugging Face dataset.")

    if args.sample_limit and not args.train_csv:
        dataset = dataset.shuffle(seed=args.seed).select(range(min(args.sample_limit, len(dataset))))

    split_1 = dataset.train_test_split(
        test_size=args.test_size,
        seed=args.seed,
        stratify_by_column="label",
    )
    split_2 = split_1["test"].train_test_split(
        test_size=0.5,
        seed=args.seed,
        stratify_by_column="label",
    )

    return DatasetDict(
        {
            "train": split_1["train"],
            "validation": split_2["train"],
            "test": split_2["test"],
        }
    )


def tokenize_dataset(dataset, tokenizer, max_length):
    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=max_length)

    return dataset.map(tokenize, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


def build_training_args(args):
    kwargs = {
        "output_dir": str(args.output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "logging_steps": 25,
        "report_to": "none",
        "seed": args.seed,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="distilbert-base-uncased")
    parser.add_argument("--dataset_name", default="atrost/financial_phrasebank")
    parser.add_argument("--dataset_config", default="")
    parser.add_argument("--train_csv", default="")
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--label_column", default="label")
    parser.add_argument("--output_dir", default="Fine-tuned_Model_files/finnews_distilbert_sentiment")
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hub_model_id", default="")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN", ""))
    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)
    set_seed(args.seed)

    dataset = load_project_dataset(args)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    tokenized = tokenize_dataset(dataset, tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    trainer_kwargs = {
        "model": model,
        "args": build_training_args(args),
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
    validation_metrics = trainer.evaluate(tokenized["validation"])
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    dataset_summary = {
        "base_model": args.base_model,
        "dataset_name": args.dataset_name if not args.train_csv else args.train_csv,
        "dataset_config": args.dataset_config if not args.train_csv else "",
        "labels": LABEL_NAMES,
        "num_train_samples": len(dataset["train"]),
        "num_validation_samples": len(dataset["validation"]),
        "num_test_samples": len(dataset["test"]),
    }
    save_json(args.output_dir / "dataset_summary.json", dataset_summary)
    save_json(args.output_dir / "metrics.json", {"validation": validation_metrics, "test": test_metrics})
    save_json(args.output_dir / "label_mapping.json", {"label2id": LABEL2ID, "id2label": ID2LABEL})

    sample_pipe = pipeline(
        "text-classification",
        model=str(args.output_dir),
        tokenizer=str(args.output_dir),
    )
    sample_text = "Tesla shares fell after weaker-than-expected deliveries and rising competition in China."
    print("\nSample prediction:")
    print(sample_text)
    print(sample_pipe(sample_text))
    print("\nSaved fine-tuned model to:", args.output_dir)
    print("Validation metrics:", validation_metrics)
    print("Test metrics:", test_metrics)

    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--hub_model_id is required when --push_to_hub is used.")
        if not args.hf_token:
            raise ValueError("HF token is required. Set HF_TOKEN or pass --hf_token.")

        api = HfApi(token=args.hf_token)
        api.create_repo(repo_id=args.hub_model_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(args.output_dir),
            repo_id=args.hub_model_id,
            repo_type="model",
            commit_message="Upload fine-tuned financial sentiment model",
        )
        print("Uploaded model to:", f"https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()
