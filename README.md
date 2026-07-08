# Twitter Sentiment Dashboard

This repository contains an NLP sentiment-analysis project based on a BERT
fine-tuning workflow. The notebook trains a transformer sequence classifier for
positive, negative, and neutral social-media sentiment.

The hosted Streamlit app is a portfolio-ready interactive dashboard. It lets
visitors:

- analyze a single tweet/review-style text
- compare multiple texts in a batch view
- inspect positive and negative word signals
- understand the original BERT training workflow
- see what model artifacts are required for real BERT inference online

## Streamlit App

Run locally:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

For Streamlit Cloud:

```text
Main file path: app.py
```

## Important Model Note

The original `App.py` loaded a fine-tuned BERT model from a local Windows path:

```text
C:\Users\Tarek Hesham\Sent_model
```

Those trained model/tokenizer artifacts are not included in this repository, so
the hosted dashboard uses a transparent lexical analyzer instead of pretending
to run unavailable BERT weights.

The original BERT app code is preserved as:

```text
legacy_bert_app.py
```

## Project Files

```text
.
|-- app.py                    # Streamlit Cloud dashboard
|-- legacy_bert_app.py        # Original BERT inference app
|-- fine-tune-bert-model.ipynb # BERT fine-tuning notebook
|-- requirements.txt          # Lightweight app dependencies
`-- README.md
```

## BERT Workflow

The notebook covers:

1. Loading a Twitter sentiment dataset.
2. Cleaning and preparing tweet text.
3. Encoding positive, negative, and neutral labels.
4. Tokenizing text with `bert-base-uncased`.
5. Fine-tuning `TFBertForSequenceClassification`.
6. Evaluating the model on a test split.
7. Saving tokenizer/model artifacts for deployment.

## Next Improvements

- Add the trained BERT tokenizer/model files or a reliable hosted model path.
- Replace the lexical analyzer with cached Hugging Face inference.
- Add test-set metrics directly to the Streamlit dashboard.
- Add confusion matrix and per-class precision/recall/F1.
- Add example social-media monitoring use cases.
