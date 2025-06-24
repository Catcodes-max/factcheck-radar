# FactCheck Radar

This project applies machine learning to identify potential misinformation in online news content. It uses semantic embeddings and classification models to generate a probability-based risk score.

## Project Overview

This tool analyzes online articles and predicts the likelihood that they contain misinformation. The model was trained on a labeled dataset of political statements and refined into a binary classification: real vs. misinformation.

## Approach

- Used a labeled dataset with categories like "true", "mostly true", and various levels of falsehood.
- Grouped labels into a binary classification (real vs. misinformation).
- Applied the MiniLM model from SentenceTransformers to embed statements.
- Trained a logistic regression classifier with class weighting to address imbalance.
- Built a second classifier using encoded news source metadata.
- Output risk scores based on predicted probabilities for misinformation.

## Model Performance (Test Set)

| Metric       | Score     |
|--------------|-----------|
| Accuracy     | 61.5%     |
| Precision    | 55.2%     |
| Recall       | 62.0%     |
| F1 Score     | 58.4%     |
| ROC AUC      | 64.6%     |


## Deployment

The model is deployed through a Streamlit app, accessible via a dropdown menu with curated article examples.

The app:
- Accepts a selected article.
- Extracts article content and computes a misinformation probability score.
- Displays prediction

## Next Steps

- Enable URL-based article extraction (pending reliable content parsing).
- Add ensemble modeling to improve precision.
- Experiment with fine-tuning MiniLM or alternative transformer-based models.
- Introduce explainability features to increase model transparency.
