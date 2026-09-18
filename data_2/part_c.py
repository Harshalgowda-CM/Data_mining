import pandas as pd
import numpy as np
import glob
import os
import re
import time
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ================================================================
# QUESTION 2(c) - SUBLINEAR RETRIEVAL AND RISK
# ================================================================

print("=" * 75)
print("QUESTION 2(c) - SUBLINEAR RETRIEVAL AND PRICE RISK")
print("=" * 75)


# ----------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------

HASH_BITS = 12

# Number of neighbouring hash buckets searched
PROBE_RADIUS = 1

# Maximum number of candidates retained for each notice
CANDIDATE_SIZE = 50

# Threshold selected in Part (A)
SIMILARITY_THRESHOLD = 0.60

# Business cost assumption:
# False merge is considered 10 times more costly than
# failing to merge a true duplicate.
FALSE_MERGE_COST = 10
MISSED_DUPLICATE_COST = 1

print("\nRetrieval settings:")
print("LSH hash bits:", HASH_BITS)
print("Probe radius:", PROBE_RADIUS)
print("Maximum candidates per notice:", CANDIDATE_SIZE)
print("Similarity threshold:", SIMILARITY_THRESHOLD)

print("\nAsymmetric error-cost setting:")
print("False merge cost:", FALSE_MERGE_COST)
print("Missed duplicate cost:", MISSED_DUPLICATE_COST)
print(
    "Cost ratio (false merge : missed duplicate):",
    f"{FALSE_MERGE_COST}:{MISSED_DUPLICATE_COST}"
)


# ----------------------------------------------------------------
# 1. LOAD NOTICES
# ----------------------------------------------------------------

notice_files = glob.glob(
    os.path.join(".", "notices", "*.csv")
)

notice_list = []

for file in sorted(notice_files):
    temp = pd.read_csv(file)
    notice_list.append(temp)

notices = pd.concat(
    notice_list,
    ignore_index=True
)

print("\nTotal notices:", len(notices))


# ----------------------------------------------------------------
# 2. CLEAN TEXT
# ----------------------------------------------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"https?://\S+|www\.\S+",
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


notices["title_clean"] = (
    notices["title"]
    .fillna("")
    .apply(clean_text)
)

notices["body_clean"] = (
    notices["body"]
    .fillna("")
    .apply(clean_text)
)

notices["text"] = (
    notices["title_clean"]
    + " "
    + notices["body_clean"]
)


# ----------------------------------------------------------------
# 3. TF-IDF REPRESENTATION
# ----------------------------------------------------------------

print("\n")
print("=" * 75)
print("BUILDING TF-IDF REPRESENTATION")
print("=" * 75)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(
    notices["text"]
)

print(
    "TF-IDF matrix shape:",
    tfidf_matrix.shape
)


# ----------------------------------------------------------------
# 4. CREATE RANDOM HYPERPLANES
# ----------------------------------------------------------------

print("\n")
print("=" * 75)
print("BUILDING LSH INDEX")
print("=" * 75)

np.random.seed(42)

dimension = tfidf_matrix.shape[1]

random_planes = np.random.randn(
    dimension,
    HASH_BITS
)

# Calculate binary hash signature
hash_values = tfidf_matrix.dot(
    random_planes
)

binary_hashes = (
    hash_values >= 0
).astype(np.uint8)


# Convert each binary signature into an integer
powers = (
    2 ** np.arange(HASH_BITS)
)

hash_keys = (
    binary_hashes * powers
).sum(axis=1)


# ----------------------------------------------------------------
# 5. BUILD BUCKET INDEX
# ----------------------------------------------------------------

buckets = {}

for index, key in enumerate(hash_keys):

    key = int(key)

    if key not in buckets:
        buckets[key] = []

    buckets[key].append(index)


print("Number of LSH buckets:", len(buckets))

bucket_sizes = np.array(
    [len(v) for v in buckets.values()]
)

print(
    "Average bucket size:",
    f"{bucket_sizes.mean():.2f}"
)

print(
    "Largest bucket size:",
    bucket_sizes.max()
)


# ----------------------------------------------------------------
# 6. FUNCTION TO GENERATE PROBE KEYS
# ----------------------------------------------------------------

def generate_probe_keys(key, bits, radius):

    keys = {int(key)}

    if radius >= 1:

        for i in range(bits):

            keys.add(
                int(key) ^ (1 << i)
            )

    if radius >= 2:

        for i in range(bits):

            for j in range(i + 1, bits):

                keys.add(
                    int(key)
                    ^ (1 << i)
                    ^ (1 << j)
                )

    return keys


# ----------------------------------------------------------------
# 7. CANDIDATE GENERATION
# ----------------------------------------------------------------

def get_candidates(index):

    key = int(hash_keys[index])

    probe_keys = generate_probe_keys(
        key,
        HASH_BITS,
        PROBE_RADIUS
    )

    candidates = set()

    for probe_key in probe_keys:

        if probe_key in buckets:

            candidates.update(
                buckets[probe_key]
            )

    # Remove query itself
    candidates.discard(index)

    # Limit candidate list
    if len(candidates) > CANDIDATE_SIZE:

        candidate_array = np.array(
            list(candidates)
        )

        query_vector = tfidf_matrix[index]

        candidate_matrix = tfidf_matrix[
            candidate_array
        ]

        similarities = cosine_similarity(
            query_vector,
            candidate_matrix
        )[0]

        order = np.argsort(
            similarities
        )[::-1]

        candidate_array = candidate_array[
            order[:CANDIDATE_SIZE]
        ]

        candidates = set(
            candidate_array.tolist()
        )

    return candidates


# ----------------------------------------------------------------
# 8. MEASURE CANDIDATE WORK
# ----------------------------------------------------------------

print("\n")
print("=" * 75)
print("MEASURING CANDIDATE RETRIEVAL")
print("=" * 75)

sample_size = min(
    len(notices),
    12000
)

candidate_counts = []

start_time = time.perf_counter()

for index in range(sample_size):

    candidates = get_candidates(index)

    candidate_counts.append(
        len(candidates)
    )

retrieval_time = (
    time.perf_counter()
    - start_time
)

candidate_counts = np.array(
    candidate_counts
)

print(
    "Notices processed:",
    sample_size
)

print(
    "Average candidates per notice:",
    f"{candidate_counts.mean():.2f}"
)

print(
    "Median candidates:",
    np.median(candidate_counts)
)

print(
    "Maximum candidates:",
    candidate_counts.max()
)

print(
    "Total candidate comparisons:",
    candidate_counts.sum()
)

full_comparisons = (
    sample_size
    * (sample_size - 1)
)

reduction = (
    1
    - candidate_counts.sum()
    / full_comparisons
) * 100

print(
    "Full-corpus comparisons:",
    full_comparisons
)

print(
    "Comparison reduction:",
    f"{reduction:.4f}%"
)

print(
    "Candidate retrieval time:",
    f"{retrieval_time:.4f} seconds"
)


# ----------------------------------------------------------------
# 9. LOAD LABELLED PAIRS
# ----------------------------------------------------------------

pairs = pd.read_csv(
    "./labelled_pairs.csv"
)

notice_index = {
    str(notices.iloc[i]["notice_id"]): i
    for i in range(len(notices))
}


# ----------------------------------------------------------------
# 10. EXACT SIMILARITY AND CANDIDATE SURVIVAL
# ----------------------------------------------------------------

pair_records = []

for _, row in pairs.iterrows():

    id_a = str(row["notice_id_a"])
    id_b = str(row["notice_id_b"])

    if (
        id_a not in notice_index
        or id_b not in notice_index
    ):
        continue

    index_a = notice_index[id_a]
    index_b = notice_index[id_b]

    exact_similarity = cosine_similarity(
        tfidf_matrix[index_a],
        tfidf_matrix[index_b]
    )[0][0]

    candidates = get_candidates(index_a)

    survived = (
        index_b in candidates
    )

    pair_records.append({
        "label": row["label"],
        "similarity": exact_similarity,
        "survived": survived
    })


pair_results = pd.DataFrame(
    pair_records
)

print("\n")
print("=" * 75)
print("LABELLED-PAIR RETRIEVAL RESULTS")
print("=" * 75)

print(
    "Pairs evaluated:",
    len(pair_results)
)


# ----------------------------------------------------------------
# 11. SURVIVAL PROBABILITY BY SIMILARITY RANGE
# ----------------------------------------------------------------

bins = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.01
]

labels_bins = [
    "0.0-0.1",
    "0.1-0.2",
    "0.2-0.3",
    "0.3-0.4",
    "0.4-0.5",
    "0.5-0.6",
    "0.6-0.7",
    "0.7-0.8",
    "0.8-0.9",
    "0.9-1.0"
]

pair_results["similarity_bin"] = pd.cut(
    pair_results["similarity"],
    bins=bins,
    labels=labels_bins,
    include_lowest=True
)

survival_table = (
    pair_results
    .groupby(
        "similarity_bin",
        observed=False
    )
    .agg(
        pairs=("survived", "count"),
        survived=("survived", "sum")
    )
)

survival_table["survival_probability"] = (
    survival_table["survived"]
    / survival_table["pairs"]
)

print("\nSurvival probability by true similarity:")

print(survival_table)


# ----------------------------------------------------------------
# 12. RETRIEVAL RECALL FOR TRUE DUPLICATES
# ----------------------------------------------------------------

same_pairs = pair_results[
    pair_results["label"].str.lower()
    == "same"
]

retrieval_recall = (
    same_pairs["survived"].mean()
)

print("\n")
print("=" * 75)
print("TRUE-DUPLICATE RETRIEVAL RECALL")
print("=" * 75)

print(
    "True duplicate pairs:",
    len(same_pairs)
)

print(
    "Retrieved true duplicates:",
    int(same_pairs["survived"].sum())
)

print(
    "Candidate-stage recall:",
    f"{retrieval_recall * 100:.2f}%"
)


# ----------------------------------------------------------------
# 13. ASYMMETRIC ERROR COST
# ----------------------------------------------------------------

pair_results["final_prediction"] = False

for i, row in pair_results.iterrows():

    if row["survived"]:

        pair_results.loc[
            i,
            "final_prediction"
        ] = (
            row["similarity"]
            >= SIMILARITY_THRESHOLD
        )


pair_results["actual"] = (
    pair_results["label"]
    .str.lower()
    == "same"
)


false_merges = (
    (~pair_results["actual"])
    & pair_results["final_prediction"]
).sum()

missed_duplicates = (
    pair_results["actual"]
    & (~pair_results["final_prediction"])
).sum()

total_cost = (
    false_merges * FALSE_MERGE_COST
    + missed_duplicates * MISSED_DUPLICATE_COST
)

print("\n")
print("=" * 75)
print("ASYMMETRIC ERROR COST")
print("=" * 75)

print(
    "False merges:",
    false_merges
)

print(
    "Missed true duplicates:",
    missed_duplicates
)

print(
    "False merge cost:",
    FALSE_MERGE_COST
)

print(
    "Missed duplicate cost:",
    MISSED_DUPLICATE_COST
)

print(
    "Total weighted error cost:",
    total_cost
)


# ----------------------------------------------------------------
# 14. OPERATING POINT
# ----------------------------------------------------------------

print("\n")
print("=" * 75)
print("SELECTED OPERATING POINT")
print("=" * 75)

print(
    "Candidate size:",
    CANDIDATE_SIZE
)

print(
    "Probe radius:",
    PROBE_RADIUS
)

print(
    "Candidate-stage recall:",
    f"{retrieval_recall * 100:.2f}%"
)

print(
    "Average candidates per notice:",
    f"{candidate_counts.mean():.2f}"
)

print(
    "Comparison reduction:",
    f"{reduction:.4f}%"
)

print(
    "False merge : missed duplicate cost ratio:",
    f"{FALSE_MERGE_COST}:{MISSED_DUPLICATE_COST}"
)

print(
    "Weighted error cost:",
    total_cost
)


# ----------------------------------------------------------------
# 15. PLOT SURVIVAL PROBABILITY
# ----------------------------------------------------------------

plot_table = survival_table.dropna(
    subset=["survival_probability"]
)

x_values = []

for value in plot_table.index:

    parts = str(value).split("-")

    lower = float(parts[0])

    upper = float(parts[1])

    x_values.append(
        (lower + upper) / 2
    )


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    x_values,
    plot_table["survival_probability"],
    marker="o"
)

plt.axvline(
    SIMILARITY_THRESHOLD,
    linestyle="--",
    label="Operating similarity threshold"
)

plt.axhline(
    retrieval_recall,
    linestyle=":",
    label="Overall duplicate recall"
)

plt.xlabel(
    "True cosine similarity"
)

plt.ylabel(
    "Probability pair survives candidate stage"
)

plt.title(
    "LSH Candidate Survival Probability vs True Similarity"
)

plt.legend()

plt.grid(True)

plt.savefig(
    "q2c_survival_probability.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ----------------------------------------------------------------
# 16. CANDIDATE WORK DISTRIBUTION
# ----------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    candidate_counts,
    bins=30
)

plt.xlabel(
    "Candidate count per notice"
)

plt.ylabel(
    "Number of notices"
)

plt.title(
    "Distribution of Candidate Work per Notice"
)

plt.grid(True)

plt.savefig(
    "q2c_candidate_work_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ----------------------------------------------------------------
# 17. FINAL INTERPRETATION
# ----------------------------------------------------------------

print("\n")
print("=" * 75)
print("INTERPRETATION")
print("=" * 75)

print("""
The retrieval stage uses Locality-Sensitive Hashing (LSH) so that
notices are first placed into locality-sensitive buckets.

A notice is compared only with notices retrieved from its own
bucket and nearby probe buckets instead of comparing it with the
entire corpus.

This creates a short candidate list and reduces the number of
expensive similarity comparisons.

The survival-probability experiment measures whether a labelled
pair survives the candidate stage as a function of its true
cosine similarity.

The selected operating point balances retrieval quality against
the amount of candidate work.

The error costs are asymmetric: a false merge is assigned a cost
of 10 while a missed duplicate is assigned a cost of 1.

Therefore, retrieval quality is not judged using recall alone;
the business cost of the two error types is explicitly included.
""")


print("\n")
print("=" * 75)
print("PART (C) COMPLETED")
print("=" * 75)