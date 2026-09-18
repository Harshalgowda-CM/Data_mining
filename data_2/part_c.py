import pandas as pd
import numpy as np
import glob
import os
import re
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ============================================================
# QUESTION 2(c) - SUBLINEAR RETRIEVAL
# ============================================================

DATA_PATH = "."


# ------------------------------------------------------------
# 1. LOAD NOTICES
# ------------------------------------------------------------

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

print("=" * 70)
print("QUESTION 2(c) - SUBLINEAR RETRIEVAL")
print("=" * 70)

print("\nTotal notices:", len(notices))


# ------------------------------------------------------------
# 2. CLEAN TEXT
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
# 3. TF-IDF
# ------------------------------------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(
    notices["text"]
)

print("\nTF-IDF shape:", tfidf_matrix.shape)


# ------------------------------------------------------------
# 4. LOAD LABELLED PAIRS
# ------------------------------------------------------------

pairs = pd.read_csv(
    os.path.join(DATA_PATH, "labelled_pairs.csv")
)

same_pairs = pairs[
    pairs["label"].str.lower() == "same"
].copy()

print("\nNumber of labelled SAME pairs:", len(same_pairs))


# ------------------------------------------------------------
# 5. NOTICE ID INDEX
# ------------------------------------------------------------

notice_index = {
    str(notices.iloc[i]["notice_id"]): i
    for i in range(len(notices))
}


# ------------------------------------------------------------
# 6. BUILD NEAREST NEIGHBOR INDEX
# ------------------------------------------------------------

print("\nBuilding nearest-neighbor index...")

nn = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

nn.fit(tfidf_matrix)

print("Index created.")


# ------------------------------------------------------------
# 7. TEST DIFFERENT CANDIDATE LIST SIZES
# ------------------------------------------------------------

candidate_sizes = [10, 25, 50, 100, 200, 500]

results = []


for k in candidate_sizes:

    print("\nTesting candidate size:", k)

    total_found = 0
    total_pairs = 0

    start_time = time.perf_counter()

    for _, row in same_pairs.iterrows():

        id_a = str(row["notice_id_a"])
        id_b = str(row["notice_id_b"])

        if id_a not in notice_index or id_b not in notice_index:
            continue

        index_a = notice_index[id_a]
        index_b = notice_index[id_b]

        distances, indices = nn.kneighbors(
            tfidf_matrix[index_a],
            n_neighbors=k + 1
        )

        candidate_indices = indices[0][1:]

        if index_b in candidate_indices:
            total_found += 1

        total_pairs += 1

    elapsed = time.perf_counter() - start_time

    retrieval_recall = (
        total_found / total_pairs
        if total_pairs > 0
        else 0
    )

    results.append([
        k,
        retrieval_recall,
        elapsed
    ])

    print(
        "Retrieval Recall:",
        f"{retrieval_recall:.4f}"
    )

    print(
        "Time:",
        f"{elapsed:.4f}",
        "seconds"
    )


# ------------------------------------------------------------
# 8. RESULTS TABLE
# ------------------------------------------------------------

results_df = pd.DataFrame(
    results,
    columns=[
        "candidate_size",
        "retrieval_recall",
        "time_seconds"
    ]
)


print("\n")
print("=" * 70)
print("SUBLINEAR RETRIEVAL RESULTS")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


# ------------------------------------------------------------
# 9. FIND SMALLEST CANDIDATE LIST WITH HIGH RECALL
# ------------------------------------------------------------

high_recall = results_df[
    results_df["retrieval_recall"] >= 0.95
]

if len(high_recall) > 0:

    selected = high_recall.iloc[0]

    print("\n")
    print("=" * 70)
    print("SELECTED CANDIDATE SIZE")
    print("=" * 70)

    print(
        "Candidate size:",
        int(selected["candidate_size"])
    )

    print(
        "Retrieval recall:",
        selected["retrieval_recall"]
    )

    print(
        "Time:",
        selected["time_seconds"],
        "seconds"
    )

else:

    print("\nNo candidate size achieved 95% retrieval recall.")


# ------------------------------------------------------------
# 10. FULL CORPUS COMPARISON COUNT
# ------------------------------------------------------------

N = len(notices)

print("\n")
print("=" * 70)
print("COMPARISON REDUCTION")
print("=" * 70)

print(
    "Full corpus size:",
    N
)

print(
    "Exact comparisons per query:",
    N
)

for k in candidate_sizes:

    reduction = (
        1 - k / N
    ) * 100

    print(
        f"{k} candidates -> "
        f"{reduction:.2f}% fewer comparisons"
    )


print("\n")
print("=" * 70)
print("PART (C) COMPLETED")
print("=" * 70)