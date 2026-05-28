# Fine-Tuned Model and Upload Guide

This folder contains the training script for the ISOM5240 financial sentiment model.

## Recommended Model

- Base model: `distilbert-base-uncased`
- Dataset: `atrost/financial_phrasebank`
- Task: three-class financial sentiment classification
- Labels: `Negative`, `Neutral`, `Positive`

The model is a standard DistilBERT classifier, which is much lighter than BERT-base and usually reliable on Streamlit Cloud.

## Train Locally

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch transformers datasets scikit-learn accelerate safetensors huggingface_hub sentencepiece tiktoken
.\.venv\Scripts\python.exe training\train_financial_sentiment.py
```

The trained model will be saved to:

```text
Fine-tuned_Model_files/finnews_distilbert_sentiment
```

## Train Quickly for Testing

Use a smaller sample if you only want to test the workflow:

```powershell
.\.venv\Scripts\python.exe training\train_financial_sentiment.py --sample_limit 600 --epochs 2
```

For the final report, train on the full dataset rather than the quick sample.

## Upload to Hugging Face

1. Create a Hugging Face account.
2. Go to `Settings > Access Tokens`.
3. Create a token with write permission.
4. Choose a model repo id, for example:

```text
your-hf-username/finnews-distilbert-sentiment
```

5. Upload from the command line:

```powershell
$env:HF_TOKEN="paste_your_token_here"
.\.venv\Scripts\python.exe training\train_financial_sentiment.py --hub_model_id your-hf-username/finnews-distilbert-sentiment --push_to_hub
```

If the model was already trained locally and you only want to upload the saved folder:

```powershell
.\.venv\Scripts\python.exe training\upload_fine_tuned_model.py --repo_id your-hf-username/finnews-distilbert-sentiment
```

## Connect the Uploaded Model to Streamlit

In Streamlit Cloud, open your app settings and add this secret:

```toml
SENTIMENT_MODEL_ID = "your-hf-username/finnews-distilbert-sentiment"
```

If the model is private, also add:

```toml
HF_TOKEN = "paste_your_token_here"
```

For final submission, the same model id should appear in:

- Project report Model URL
- Streamlit Cloud secrets
- GitHub README
- Fine-tuning notebook output
- Experimental_results.xlsx
