import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import re
import spacy
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
nlp.max_length = 20000000


def preprocess(text):
    """Preprocess text by removing punctuation, stopwords, and lemmatizing, and converting to lowercase"""
    doc = nlp(text.lower())  # Convert text to lowercase
    result = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and token.is_alpha
        and not token.is_punct
        and not token.like_num
    ]
    return " ".join(result)


def join_headlines(series):
    """Combine headlines in a series into a single string"""
    # Limit the number of headlines to 100 -> can be changed
    return " ".join(series.sample(n=min(100, len(series)), random_state=1))


def get_bert_embedding(text):
    """Convert text to a BERT embedding"""
    return model.encode(text)


df = pd.read_csv("headlines.csv")
df = df.dropna(subset=["Headline"])  # Remove rows with empty headlines

# Convert the "Date" column to datetime objects
df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")

publication_date_range = df.groupby("Publication")["Date"].agg(["min", "max"])

# Finding the overlap in date range
# Initialize the overlap range with the first publication's date range
min_date, max_date = publication_date_range.iloc[0]

# Iterate through the date ranges to find the overlap
for _, row in publication_date_range.iterrows():
    min_date = max(min_date, row["min"])
    max_date = min(max_date, row["max"])

df = df[df["Date"] >= min_date]
df = df[df["Date"] <= max_date]

# Find all unique publications
unique_publications = df["Publication"].unique()

# Create a dictionary to store resampled data for each publication
publication_resampled_data = {}
for pub in df["Publication"].unique():
    pub_df = df[df["Publication"] == pub].set_index("Date")
    resampled_df = pub_df.resample("6M").agg({"Headline": join_headlines})
    publication_resampled_data[pub] = resampled_df["Headline"].apply(preprocess)
