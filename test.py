import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import re
import spacy
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

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
    if pub == "USA Today":
        continue
    pub_df = df[df["Publication"] == pub].set_index("Date")
    resampled_df = pub_df.resample("6M").agg({"Headline": join_headlines})
    publication_resampled_data[pub] = resampled_df["Headline"].apply(preprocess)

# Document Similarities

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()


# Topic Modeling
topic_model = BERTopic(verbose=True)

timestamps = [
    "2009-09-30",
    "2010-03-31",
    "2010-09-30",
    "2011-03-31",
    "2011-09-30",
    "2012-03-31",
    "2012-09-30",
    "2013-03-31",
    "2013-09-30",
    "2014-03-31",
    "2014-09-30",
    "2015-03-31",
    "2015-09-30",
    "2016-03-31",
    "2016-09-30",
    "2017-03-31",
    "2017-09-30",
    "2018-03-31",
    "2018-09-30",
    "2019-03-31",
    "2019-09-30",
    "2020-03-31",
    "2020-09-30",
    "2021-03-31",
    "2021-09-30",
    "2022-03-31",
    "2022-09-30",
    "2023-03-31",
]

data = {}
for pub in publication_resampled_data:
    print(pub)
    topics, probs = topic_model.fit_transform(publication_resampled_data[pub].to_list())
    topics_over_time = topic_model.topics_over_time(
        publication_resampled_data[pub].to_list(), timestamps, nr_bins=20
    )
    data[pub] = topics_over_time


# Sentiment Analysis
