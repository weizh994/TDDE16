import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import re


def compute_cluster_summaries(kmeans, vectorizer, top_n):
    """Compute the top_n highest-weighted terms from the centroid of each cluster.

    Arguments:
        kmeans: The trained k-means classifier.
        vectorizer: The fitted vectorizer; needed to obtain the actual terms
                    belonging to the items in the cluster.
        top_n: The number of terms to return for each cluster.

    Returns:
        A list of length k, where k is the number of clusters. Each item in the list
        should be a list of length `top_n` with the highest-weighted terms from that
        cluster.  Example:
          [["first", "foo", ...], ["second", "bar", ...], ["third", "baz", ...]]
    """
    result = []
    for k in range(kmeans.n_clusters):
        top_n_idx = np.argsort(kmeans.cluster_centers_[k])[::-1][:top_n]
        temp = []
        for idx in top_n_idx:
            temp.append(vectorizer.get_feature_names_out()[idx])
        result.append(temp)
    return result


def fit_kmeans(data, n_clusters):
    """Fit a k-means classifier to some data.

    Arguments:
        data: The vectorized data to train the classifier on.
        n_clusters (int): The number of clusters.

    Returns:
        The trained k-means classifier.
    """
    return KMeans(n_clusters=n_clusters, n_init="auto").fit(data)


def plot_cluster_size(kmeans):
    """Produce & display a bar plot with the number of documents per cluster.

    Arguments:
        kmeans: The trained k-means classifier.
    """
    plt.bar(range(kmeans.n_clusters), pd.Series(kmeans.labels_).value_counts())


# Read the data
df_bbc = pd.read_csv("bbc_news.csv")
# df_cnn = pd.read_csv("cnn_news.csv")

# Convert the date to datetime
df_bbc["pubDate"] = pd.to_datetime(df_bbc["pubDate"])
# df_cnn["Date published"] = pd.to_datetime(df_cnn["Date published"])

df_bbc.set_index("pubDate", inplace=True)
# df_cnn.set_index("Date published", inplace=True)

count_bbc = df_bbc.resample("6M").count()
# count_cnn = df_cnn.resample("6M").count()


# 打印结果
print("BBC每三个月的数据计数:")
print(count_bbc)
# print("\nCNN每三个月的数据计数:")
# print(count_cnn)

# Clean the data
df_bbc["description"] = df_bbc["description"].apply(
    lambda x: re.sub(r"\W", " ", str(x))
)
# df_cnn["Description"] = df_cnn["Description"].apply(
#    lambda x: re.sub(r"\W", " ", str(x))
# )
df_bbc["description"] = df_bbc["description"].apply(lambda x: x.lower())
# df_cnn["Description"] = df_cnn["Description"].apply(lambda x: x.lower())


"""# Vectorize the data
vectorizer = TfidfVectorizer(stop_words="english")
vectorized_bbc = vectorizer.fit_transform(df_bbc["description"])
vectorized_cnn = vectorizer.fit_transform(df_cnn["Description"])

vectorized_data = vectorizer.fit_transform(
    pd.concat([df_bbc["description"], df_cnn["Description"]])
)
"""

"""from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(vectorized_bbc)

feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    message = "Topic #%d: " % topic_idx
    message += " ".join([feature_names[i] for i in topic.argsort()[: -10 - 1 : -1]])
    print(message)"""
