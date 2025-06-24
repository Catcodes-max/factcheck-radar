#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from joblib import load
from sklearn.preprocessing import LabelEncoder
import trafilatura

# -------------------------
# Load model + transformers
# -------------------------


@st.cache_resource
def load_models_and_encoders():
    # Load main text classifier
    clf_text = load("saved_model.pkl")
    
    # Load sentence transformer
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Load classifier and encoder
    clf_source = load("source_only_model.pkl")
    source_encoder = load("source_encoder.pkl")
    ohe = load("source_ohe.pkl")
    
    return clf_text, encoder, clf_source, source_encoder, ohe

clf_text, encoder, clf_source, source_encoder, ohe = load_models_and_encoders()


# -------------------------
# Streamlit App UI
# -------------------------
st.title("FactCheck Radar")
st.markdown(
    "This app demonstrates a fake news classification model trained on known misinformation sources. "
    "It evaluates the credibility of article text using sentence embeddings and a custom-trained classifier. "
    "For demonstration purposes, article content is loaded from preselected sources to ensure stability."
)

sample_articles = {
    "BBC News (Real News – Climate Report)": "A new climate report by the UN warns of record-breaking global temperatures and rising sea levels, urging immediate international action to mitigate effects.",
    "Reuters (Real News – Economic Outlook)": "The U.S. economy grew at a 2.1% annualized rate last quarter, with consumer spending and job creation remaining steady according to the Labor Department.",
    "NPR (Real News – Health Policy)": "The new healthcare bill passed in the Senate will expand access to rural medical services and fund preventative care initiatives across underserved communities.",
    "Clickbait Site (Suspicious Claim About Health)": "Scientists hate her! This one simple trick reverses aging instantly. Big Pharma doesn’t want you to know about this breakthrough.",
    "Fake News Blog (Political Misinformation)": "Breaking: Presidential candidate caught running secret alien government under the White House, claims anonymous insider.",
    "Satirical Source (Clearly Fabricated Story)": "Government officials announce new plan to replace all birds with government surveillance drones by 2026.",
}

selected_sample = st.selectbox("Choose an article to analyze:", list(sample_articles.keys()))

if selected_sample:
    full_text = sample_articles[selected_sample]
    st.success("Loaded article content successfully.")
    X_embed = encoder.encode([full_text], show_progress_bar=False)
    text_prob = clf_text.predict_proba(X_embed)[:, 1][0]
    st.metric("Fake News Probability", f"{text_prob:.2%}")
    st.markdown(f"**Full extracted text ({len(full_text)} characters):**")
    with st.expander("Show full article text"):
        st.write(full_text)
