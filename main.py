import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import re
import spacy
from sentence_transformers import SentenceTransformer
import seaborn as sns


model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])


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
        and token.pos_ != "VERB"  # Exclude verbs
    ]
    return " ".join(result)


def join_headlines(series):
    """Combine headlines in a series into a single string"""
    # Limit the number of headlines to 100 -> can be changed
    return " ".join(series.sample(n=min(500, len(series)), random_state=1))


def get_bert_embedding(text):
    """Convert text to a BERT embedding"""
    return model.encode(text)


df = pd.read_csv("headlines.csv")
df = df.dropna(subset=["Headline"])  # Remove rows with empty headlines
# Remove all rows related to "USA Today"
df = df[df["Publication"] != "USA Today"]

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
    resampled_df = pub_df.resample("1Y").agg({"Headline": join_headlines})
    publication_resampled_data[pub] = resampled_df["Headline"].apply(preprocess)


timestamps = [
    "2009-12-31",
    "2010-12-31",
    "2011-12-31",
    "2012-12-31",
    "2013-12-31",
    "2014-12-31",
    "2015-12-31",
    "2016-12-31",
    "2017-12-31",
    "2018-12-31",
    "2019-12-31",
    "2020-12-31",
    "2021-12-31",
    "2022-12-31",
]
# Document Similarities
"""from sklearn.metrics.pairwise import cosine_similarity

embeddings = {}
for pub in unique_publications:
    for timestamp in timestamps:
        embeddings[(pub, timestamp)] = get_bert_embedding(
            publication_resampled_data[pub][timestamp]
        )
similarity_matrix = {}
for pub in unique_publications:
    if pub != "BBC":
        for timestamp in timestamps:
            bbc_embedding = embeddings[("BBC", timestamp)]
            pub_embedding = embeddings[(pub, timestamp)]
            similarity = cosine_similarity([bbc_embedding], [pub_embedding])[0][0]
            similarity_matrix[(pub, timestamp)] = similarity
# Plotting the similarity scores

data = [
    {"Publication": pub, "Date": date, "Similarity": sim}
    for (pub, date), sim in similarity_matrix.items()
]

similarity_df = pd.DataFrame(data)

similarity_df["Date"] = pd.to_datetime(similarity_df["Date"]).dt.year

pivot_table = similarity_df.pivot(
    index="Publication", columns="Date", values="Similarity"
)

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, cmap="coolwarm")
plt.title("Document Similarity Heatmap with BBC")
plt.show()
"""
# Topic Modeling
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer

representation_model = KeyBERTInspired()
vectorizer_model = CountVectorizer(stop_words="english")
ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)
topic_model = BERTopic(
    language="english",
    calculate_probabilities=True,
    min_topic_size=10,
    verbose=True,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    representation_model=representation_model,
)

data = {}
sampled_df = df.sample(frac=0.1, random_state=1)

# all data

for pub in sampled_df["Publication"].unique():
    print(pub)
    pub_df = sampled_df[sampled_df["Publication"] == pub]
    documents = pub_df["Headline"].tolist()
    topics, probs = topic_model.fit_transform(documents)
    topics_over_time = topic_model.topics_over_time(
        documents, pub_df["Date"].tolist(), nr_bins=20
    )
    print(topics_over_time)
    freq = topic_model.get_topic_info()
    print(freq.head(10))
    fig = topic_model.visualize_topics()
    fig.write_html(f"figure/{pub}_all_topics.html")
    fig = topic_model.visualize_barchart(top_n_topics=5)
    fig.write_html(f"figure//{pub}_top5.html")
    fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=20)
    # data[pub] = topics_over_time
    # data[pub].to_csv(f"{pub}_topics_over_time.csv")
    fig.write_html(f"figure/{pub}_topics_over_time.html")


# resampled data with topics_over_time
"""for pub in publication_resampled_data:
    print(pub)
    topics, probs = topic_model.fit_transform(publication_resampled_data[pub].to_list())
    fig = topic_model.visualize_topics()
    fig.write_html(f"figure/{pub}_all_topics.html")
    topics_over_time = topic_model.topics_over_time(
        publication_resampled_data[pub].to_list(), timestamps, nr_bins=20
    )
    # print(topics_over_time)
    data[pub] = topics_over_time
    print(data[pub])
    #fig = topic_model.visualize_topics_over_time(topics_over_time)
    #fig.write_html(f"{pub}_topics_over_time.html")
"""
# Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

vader = {}

for pub in publication_resampled_data:
    vader[pub] = {}
    for timestamp in timestamps:
        vader[pub][timestamp] = analyzer.polarity_scores(
            publication_resampled_data[pub][timestamp]
        )["compound"]


sentiment_df = pd.DataFrame.from_dict(
    {
        (pub, timestamp): vader[pub][timestamp]
        for pub in vader.keys()
        for timestamp in vader[pub].keys()
    },
    orient="index",
)


# Plotting the sentiment scores
sentiment_df = sentiment_df.reset_index()
sentiment_df[["Publication", "Date"]] = pd.DataFrame(
    sentiment_df["index"].tolist(), index=sentiment_df.index
)
sentiment_df.rename(columns={0: "Sentiment"}, inplace=True)
sentiment_df.drop(columns=["index"], inplace=True)
sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"]).dt.year

pivot_table = sentiment_df.pivot(
    index="Publication", columns="Date", values="Sentiment"
)

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, cmap="coolwarm")
plt.title("Sentiment Heatmap")
plt.show()
