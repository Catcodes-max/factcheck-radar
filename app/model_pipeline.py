#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



# In[11]:


# Define column names
columns = [
    "id", "label", "statement", "subject", "speaker", "speaker_job",
    "speaker_state", "party", "barely_true_counts", "false_counts",
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts",
    "context"
]

# Load each file 

test_df = pd.read_csv("/Users/catherineakins/Desktop/Projects/MLFactChecker/Data/test.tsv", sep='\t', header=None, names=columns)
train_df = pd.read_csv("/Users/catherineakins/Desktop/Projects/MLFactChecker/Data/train.tsv", sep='\t', header=None, names=columns, index_col=False)
val_df = pd.read_csv("/Users/catherineakins/Desktop/Projects/MLFactChecker/Data/valid.tsv", sep='\t', header=None, names=columns,index_col=False)

####sim source data######
source_pool = [
    "nytimes.com", "cnn.com", "theonion.com", "infowars.com", "reuters.com",
    "breitbart.com", "foxnews.com", "naturalnews.com", "bbc.com"
]

np.random.seed(42)
train_df["source"] = np.random.choice(source_pool, size=len(train_df))
val_df["source"] = np.random.choice(source_pool, size=len(val_df))
test_df["source"] = np.random.choice(source_pool, size=len(test_df))

# Preview
print(train_df.head())
print(train_df["label"].value_counts())


# In[12]:


true_labels = ["true", "mostly-true", "half-true"]

for df in [train_df, val_df, test_df]:
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["is_misinformation"] = df["label"].apply(lambda x: 0 if x in true_labels else 1)


# In[13]:


for name, df in zip(["Train", "Validation", "Test"], [train_df, val_df, test_df]):
    print(f"{name} set label counts:")
    print(df["is_misinformation"].value_counts())
    print()










model = SentenceTransformer('all-MiniLM-L6-v2')  
statements = df["statement"].tolist()
X_text = model.encode(statements, show_progress_bar=True)

X_train_combined = model.encode(train_df["statement"].tolist(), show_progress_bar=True)
X_val_combined = model.encode(val_df["statement"].tolist(), show_progress_bar=True)
X_test_combined = model.encode(test_df["statement"].tolist(), show_progress_bar=True)


# In[15]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_train = train_df["is_misinformation"]
y_val = val_df["is_misinformation"]
y_test = test_df["is_misinformation"]
clf = LogisticRegression(class_weight='balanced', solver='saga', penalty='l1')

clf.fit(X_train_combined, y_train)
y_pred = clf.predict(X_test_combined)

# Predict class probabilities for ROC-AUC (optional)
y_proba = clf.predict_proba(X_test_combined)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

# Display
print("=== Model Performance on Test Set ===")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC AUC   : {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))







# Encode source
source_encoder = LabelEncoder()
train_df["source_encoded"] = source_encoder.fit_transform(train_df["source"])
val_df["source_encoded"] = source_encoder.transform(val_df["source"])
test_df["source_encoded"] = source_encoder.transform(test_df["source"])

# One-hot encode source
ohe = OneHotEncoder(handle_unknown="ignore")
X_source_train = ohe.fit_transform(train_df[["source"]])
X_source_val = ohe.transform(val_df[["source"]])
X_source_test = ohe.transform(test_df[["source"]])

# Train classifier
clf_source = LogisticRegression(class_weight="balanced", max_iter=1000)
clf_source.fit(X_source_train, y_train)



# Get probability that post is misinformation (class 1)
y_proba = clf.predict_proba(X_test_combined)[:, 1]





test_df["P_x"] = y_proba
test_df["Risk"] = test_df["P_x"]





N = 20
top_posts = test_df.sort_values("Risk", ascending=False).head(N)





display_cols = ["statement", "label", "Risk"]
print(top_posts[display_cols])



dump(clf, "saved_model.pkl")
dump(clf_source, "source_only_model.pkl")
dump(source_encoder, "source_encoder.pkl")  # this is the LabelEncoder
dump(ohe, "source_ohe.pkl")  # this is the OneHotEncoder 


sample_articles = {
    "BBC News (Real News – Climate Report)": "A new climate report by the UN warns of record-breaking global temperatures and rising sea levels, urging immediate international action to mitigate effects.",
    "Reuters (Real News – Economic Outlook)": "The U.S. economy grew at a 2.1% annualized rate last quarter, with consumer spending and job creation remaining steady according to the Labor Department.",
    "NPR (Real News – Health Policy)": "The new healthcare bill passed in the Senate will expand access to rural medical services and fund preventative care initiatives across underserved communities.",
}
