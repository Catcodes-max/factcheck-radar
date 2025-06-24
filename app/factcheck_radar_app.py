#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
from joblib import load
import numpy as np


# In[4]:


# Load the trained classifier
clf = load("saved_model.pkl")

# Load the vectorizer and SVD transformer if they were saved
vectorizer = load("saved_vectorizer.pkl")
svd = load("saved_svd.pkl")


# In[5]:


# Load the test dataset
test_df = pd.read_csv("test_simulated.csv")


# In[6]:


st.title("🧠 FactCheck Radar")
st.subheader("Prioritizing content for review using ML + Risk Scoring")


# In[7]:


st.sidebar.header("Filters")
top_n = st.sidebar.slider("Number of posts to review", min_value=5, max_value=50, value=20)


# In[8]:


# Vectorize the statements
X_test_text = vectorizer.transform(test_df["statement"])

# Apply SVD transformation
X_test_pca = svd.transform(X_test_text)

# Combine with metadata
X_test_meta = test_df[["likes", "shares", "source_reach"]].values
X_test_combined = np.hstack([X_test_pca, X_test_meta])

# Predict probabilities
test_df["P(x)"] = clf.predict_proba(X_test_combined)[:, 1]

# Calculate Spread Score (S(x))
test_df["S(x)"] = (
    test_df["likes"] * 0.3 +
    test_df["shares"] * 0.5 +
    test_df["source_reach"] * 0.2
)

# Compute Risk Score
test_df["Risk"] = test_df["P(x)"] * test_df["S(x)"]


# In[9]:


st.subheader(f"Top {top_n} Posts by Risk Score")
top_posts = test_df.sort_values("Risk", ascending=False).head(top_n)
st.dataframe(top_posts[["statement", "P(x)", "S(x)", "Risk", "likes", "shares", "source_reach"]])


# In[ ]:




