# Financial News Sentiment & Risk Briefing Assistant

This is the Streamlit app code for the ISOM5240 project:

**Financial News Sentiment & Risk Briefing Assistant for Beginner Investors**

The app uses two Hugging Face pipelines:

1. Financial sentiment classification
2. Named entity recognition

It does not provide buy, sell, or hold recommendations.

## Files

```text
app.py
utils.py
requirements.txt
sample_inputs.csv
```

## Local run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Model configuration

By default, the app uses:

```text
SENTIMENT_MODEL_ID=mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
NER_MODEL_ID=dslim/bert-base-NER
```

For the final submission, set `SENTIMENT_MODEL_ID` to your group's fine-tuned Hugging Face model id.
The app will show it as **Fine-tuned Sentiment Model** in the model selector.

On Streamlit Cloud, add this in **Secrets**:

```toml
SENTIMENT_MODEL_ID="your-huggingface-username/your-finetuned-finbert-model"
NER_MODEL_ID="dslim/bert-base-NER"
```

The same sentiment model id should appear in:

- app.py or Streamlit secrets
- Project report
- Experimental_results.xlsx
- Hugging Face model URL
- Fine-tuning notebook output

## Fine-tuning

Fine-tuning materials are in:

```text
training/
```

Recommended notebook:

```text
training/fine_tune_financial_sentiment_colab.ipynb
```

Upload instructions:

```text
training/README_finetune_upload.md
```

## Batch testing CSV format

Use `sample_inputs.csv` as the template.

Required:

```text
text
```

Optional:

```text
expected_label
```

Accepted labels:

```text
Positive, Neutral, Negative
```

If `expected_label` is included, the app calculates app testing accuracy.

## Coursework note

The risk briefing section uses transparent keyword rules. The deep learning part of the project is the fine-tuned financial sentiment classifier, while the second Hugging Face pipeline is named entity recognition.
