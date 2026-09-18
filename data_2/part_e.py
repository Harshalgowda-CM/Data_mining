import pandas as pd
import numpy as np
import glob
import os
import re
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ============================================================
# QUESTION 2(e) - FULL CORPUS EXPERIMENT
# ============================================================

DATA_PATH = "."

print("=" * 70)
print("QUESTION 2(e) - FULL CORPUS EXPERIMENT")
print("=" * 70)


# ------------------------------------------------------------
# 1. LOAD FULL CORPUS
# ------------------------------------------------------------

start_total = time.perf_counter()

notice_files = glob.glob(
    os.path.join(DATA_PATH, "notices", "*.csv")
)

notice_list = []

for file in notice_files:
    notice_list.append(pd.read_csv(file))

notices = pd.concat(
    notice_list,
    ignore_index=True
)

print("\nTotal notices:", len(notices))


# ------------------------------------------------------------
# 2. TEXT CLEANING
# ------------------------------------------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


notices["text"] = (
    notices["title"].fillna("").apply(clean_text)
    + " "
    + notices["body"].fillna("").apply(clean_text)
)


# ------------------------------------------------------------
# 3. BEFORE MITIGATION
#    Full 5000-dimensional representation
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("BEFORE MITIGATION")
print("=" * 70)

start_before = time.perf_counter()

vectorizer_full = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

tfidf_full = vectorizer_full.fit_transform(
    notices["text"]
)

nn_full = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

nn_full.fit(tfidf_full)

before_time = time.perf_counter() - start_before

print(
    "Representation shape:",
    tfidf_full.shape
)

print(
    "Before-mitigation runtime:",
    f"{before_time:.4f}",
    "seconds"
)


# ------------------------------------------------------------
# 4. IDENTIFY SKEW / FAILURE
# ------------------------------------------------------------

pairs = pd.read_csv(
    os.path.join(DATA_PATH, "labelled_pairs.csv")
)

same_pairs = pairs[
    pairs["label"].str.lower() == "same"
].copy()

notice_index = {
    str(notices.iloc[i]["notice_id"]): i
    for i in range(len(notices))
}


print("\n")
print("=" * 70)
print("FAILURE / SKEW ANALYSIS")
print("=" * 70)

print(
    "Total labelled same pairs:",
    len(same_pairs)
)

print(
    "Class distribution:"
)

print(
    pairs["label"].value_counts()
)

# Portal distribution
portal_counts = notices["portal_id"].value_counts()

print(
    "\nNumber of portals:",
    notices["portal_id"].nunique()
)

print(
    "Largest portal size:",
    portal_counts.max()
)

print(
    "Smallest portal size:",
    portal_counts.min()
)

print(
    "Average notices per portal:",
    portal_counts.mean()
)


# ------------------------------------------------------------
# 5. MITIGATION
#    Reduce representation to 100 SVD dimensions
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("AFTER MITIGATION")
print("=" * 70)

start_after = time.perf_counter()

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(
    n_components=100,
    random_state=42
)

reduced = svd.fit_transform(
    tfidf_full
)

nn_reduced = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

nn_reduced.fit(reduced)

after_time = time.perf_counter() - start_after

print(
    "Reduced representation shape:",
    reduced.shape
)

print(
    "Explained variance:",
    svd.explained_variance_ratio_.sum()
)

print(
    "After-mitigation runtime:",
    f"{after_time:.4f}",
    "seconds"
)


# ------------------------------------------------------------
# 6. RUNTIME COMPARISON
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("RUNTIME COMPARISON")
print("=" * 70)

print(
    "Before mitigation:",
    f"{before_time:.4f}",
    "seconds"
)

print(
    "After mitigation:",
    f"{after_time:.4f}",
    "seconds"
)

if after_time < before_time:

    improvement = (
        (before_time - after_time)
        / before_time
    ) * 100

    print(
        "Runtime reduction:",
        f"{improvement:.2f}%"
    )

else:

    increase = (
        (after_time - before_time)
        / before_time
    ) * 100

    print(
        "Runtime change:",
        f"{increase:.2f}% increase"
    )


# ------------------------------------------------------------
# 7. 20-MINUTE NIGHTLY BUDGET
# ------------------------------------------------------------

budget_seconds = 20 * 60

print("\n")
print("=" * 70)
print("20-MINUTE NIGHTLY BUDGET")
print("=" * 70)

print(
    "Budget:",
    budget_seconds,
    "seconds"
)

if before_time <= budget_seconds:
    print("Before mitigation: WITHIN BUDGET")
else:
    print("Before mitigation: EXCEEDS BUDGET")

if after_time <= budget_seconds:
    print("After mitigation: WITHIN BUDGET")
else:
    print("After mitigation: EXCEEDS BUDGET")


print("\n")
print("=" * 70)
print("PART (E) COMPLETED")
print("=" * 70)