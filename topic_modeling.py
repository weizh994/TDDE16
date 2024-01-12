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

df = pd.read_csv("headlines.csv")
df = df[df["Publication"] != "USA Today"]
pubs = []
pubs = df["Publication"].unique()
for pub in pubs:
    print(pub)
    df = pd.read_csv(f"{pub}_topics_over_time.csv")
    # Convert 'Timestamp' to datetime and extract the year
    df["Year"] = pd.to_datetime(df["Timestamp"], errors="coerce").dt.year
    top_freq_per_year = (
        df.groupby("Year")
        .apply(lambda x: x.nlargest(1, "Frequency"))
        .reset_index(drop=True)
    )
    print(top_freq_per_year)
