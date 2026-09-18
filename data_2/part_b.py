import pandas as pd
import numpy as np
import glob
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ================================================================
# QUESTION 2(b) - TRADE EXACTNESS FOR SPACE
# ================================================================

print("=" * 75)
print("QUESTION 2(b) - TRADE EXACTNESS FOR SPACE")
print("=" * 75)


# ----------------------------------------------------------------
# 1. LOAD NOTICES
# ----------------------------------------------------------------

notice_files = glob.glob(os.path.join(".", "notices", "*.csv"))

notice_list = []

for file in sorted(notice_files):
    temp = pd.read_csv(file)
    notice_list.append(temp)

notices = pd.concat(notice_list, ignore_index=True)

print("\nTotal notices:", len(notices))


# ----------------------------------------------------------------
# 2. TEXT CLEANING
# ----------------------------------------------------------------

def clean_text(text):

    text = str(text).lower()

    # Remove HTML
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

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

print("Text preprocessing completed.")


# ----------------------------------------------------------------
# 3. LOAD TRUSTED LABELLED PAIRS
# ----------------------------------------------------------------

pairs = pd.read_csv("./labelled_pairs.csv")

print("\nNumber of labelled pairs:", len(pairs))

print("\nLabel distribution:")
print(pairs["label"].value_counts())


# ----------------------------------------------------------------
# 4. BUILD NOTICE ID INDEX
# ----------------------------------------------------------------

notice_index = {
    str(notices.iloc[i]["notice_id"]): i
    for i in range(len(notices))
}


# ----------------------------------------------------------------
# 5. FULL TF-IDF REPRESENTATION
# ----------------------------------------------------------------

print("\n")
print("=" * 75)
print("FULL REPRESENTATION")
print("=" * 75)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(
    notices["text"]
)

print("Full TF-IDF shape:", tfidf_matrix.shape)


# ----------------------------------------------------------------
# 6. REDUCED REPRESENTATION
# ----------------------------------------------------------------
#
# We deliberately reduce 5000 dimensions to 100 dimensions.
#
# Reason:
# The application needs a much smaller representation while
# retaining enough information for approximate similarity.
#
# ----------------------------------------------------------------

REDUCED_DIMENSIONS = 100

print("\n")
print("=" * 75)
print("REDUCED REPRESENTATION")
print("=" * 75)

svd = TruncatedSVD(
    n_components=REDUCED_DIMENSIONS,
    random_state=42
)

reduced_matrix = svd.fit_transform(
    tfidf_matrix
)

print(
    "Reduced representation shape:",
    reduced_matrix.shape
)

explained_variance = svd.explained_variance_ratio_.sum()

print(
    "Explained variance:",
    explained_variance
)


# ----------------------------------------------------------------
# 7. SPACE REDUCTION
# ----------------------------------------------------------------

original_dimensions = tfidf_matrix.shape[1]
reduced_dimensions = reduced_matrix.shape[1]

space_reduction = (
    1 - reduced_dimensions / original_dimensions
) * 100

print("\n")
print("=" * 75)
print("SPACE REDUCTION")
print("=" * 75)

print("Original dimensions:", original_dimensions)
print("Reduced dimensions:", reduced_dimensions)

print(
    "Dimensionality reduction:",
    f"{space_reduction:.2f}%"
)

print(
    "Reduction ratio:",
    f"{original_dimensions / reduced_dimensions:.1f}x"
)


# ----------------------------------------------------------------
# 8. CALCULATE EXACT VS APPROXIMATE SIMILARITY
# ----------------------------------------------------------------

exact_scores = []
approximate_scores = []
labels = []


for _, row in pairs.iterrows():

    id_a = str(row["notice_id_a"])
    id_b = str(row["notice_id_b"])

    if id_a not in notice_index or id_b not in notice_index:
        continue

    index_a = notice_index[id_a]
    index_b = notice_index[id_b]

    # Exact similarity using full TF-IDF
    exact_score = cosine_similarity(
        tfidf_matrix[index_a],
        tfidf_matrix[index_b]
    )[0][0]

    # Approximate similarity using reduced representation
    vector_a = reduced_matrix[index_a]
    vector_b = reduced_matrix[index_b]

    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )

    if denominator == 0:
        approximate_score = 0
    else:
        approximate_score = (
            np.dot(vector_a, vector_b)
            / denominator
        )

    exact_scores.append(exact_score)
    approximate_scores.append(approximate_score)
    labels.append(row["label"])


exact_scores = np.array(exact_scores)
approximate_scores = np.array(approximate_scores)


# ----------------------------------------------------------------
# 9. ESTIMATION ERROR
# ----------------------------------------------------------------

absolute_errors = np.abs(
    exact_scores - approximate_scores
)

mae = mean_absolute_error(
    exact_scores,
    approximate_scores
)

rmse = np.sqrt(
    mean_squared_error(
        exact_scores,
        approximate_scores
    )
)

max_error = np.max(absolute_errors)

print("\n")
print("=" * 75)
print("SIMILARITY ESTIMATION ERROR")
print("=" * 75)

print("Pairs evaluated:", len(exact_scores))

print(
    "Mean Absolute Error (MAE):",
    f"{mae:.6f}"
)

print(
    "Root Mean Squared Error (RMSE):",
    f"{rmse:.6f}"
)

print(
    "Maximum absolute error:",
    f"{max_error:.6f}"
)


# ----------------------------------------------------------------
# 10. ERROR BY LABEL
# ----------------------------------------------------------------

results = pd.DataFrame({
    "label": labels,
    "exact_similarity": exact_scores,
    "approximate_similarity": approximate_scores,
    "absolute_error": absolute_errors
})

print("\n")
print("=" * 75)
print("ERROR BY LABEL")
print("=" * 75)

print(
    results
    .groupby("label")["absolute_error"]
    .agg(
        ["count", "mean", "max"]
    )
)


# ----------------------------------------------------------------
# 11. AVERAGE SIMILARITY
# ----------------------------------------------------------------

print("\n")
print("=" * 75)
print("AVERAGE SIMILARITY")
print("=" * 75)

print("\nExact TF-IDF:")

print(
    results
    .groupby("label")["exact_similarity"]
    .mean()
)

print("\nReduced representation:")

print(
    results
    .groupby("label")["approximate_similarity"]
    .mean()
)


# ----------------------------------------------------------------
# 12. THRESHOLD TEST
# ----------------------------------------------------------------
#
# The threshold from Part (A) was 0.60 for the full representation.
#
# We apply the same threshold to the reduced representation and
# measure how many decisions change.
#
# ----------------------------------------------------------------

THRESHOLD = 0.60

results["exact_prediction"] = (
    results["exact_similarity"] >= THRESHOLD
)

results["approximate_prediction"] = (
    results["approximate_similarity"] >= THRESHOLD
)

decision_changes = (
    results["exact_prediction"]
    != results["approximate_prediction"]
).sum()

decision_change_percentage = (
    decision_changes / len(results)
) * 100


print("\n")
print("=" * 75)
print("DECISION STABILITY")
print("=" * 75)

print("Similarity threshold:", THRESHOLD)

print(
    "Changed decisions:",
    decision_changes
)

print(
    "Decision change percentage:",
    f"{decision_change_percentage:.2f}%"
)


# ----------------------------------------------------------------
# 13. ESTIMATION ACCURACY
# ----------------------------------------------------------------

decision_accuracy = (
    results["exact_prediction"]
    == results["approximate_prediction"]
).mean()

print(
    "Decision agreement:",
    f"{decision_accuracy * 100:.2f}%"
)


# ----------------------------------------------------------------
# 14. REPRESENTATION SIZE COMPARISON
# ----------------------------------------------------------------

print("\n")
print("=" * 75)
print("REPRESENTATION COMPARISON")
print("=" * 75)

print(
    f"Full representation : {original_dimensions} dimensions"
)

print(
    f"Reduced representation : {reduced_dimensions} dimensions"
)

print(
    f"Space reduction : {space_reduction:.2f}%"
)

print(
    f"Explained variance : {explained_variance:.4f}"
)

print(
    f"Similarity MAE : {mae:.6f}"
)

print(
    f"Decision agreement : {decision_accuracy * 100:.2f}%"
)


# ----------------------------------------------------------------
# 15. JUDGEMENT
# ----------------------------------------------------------------

print("\n")
print("=" * 75)
print("JUDGEMENT")
print("=" * 75)

print("""
The full TF-IDF representation provides the exact similarity
calculation but requires 5000 dimensions per notice.

The reduced representation uses only 100 dimensions.
This gives a 98% dimensionality reduction while retaining
approximately the measured explained variance reported above.

The reduced representation is therefore treated as an
approximate similarity representation rather than an exact one.

The final decision is based on the measured similarity error
and decision agreement on the 900 trusted labelled pairs.

This closes the loop between the expected space saving and the
realised estimation error.
""")


# ----------------------------------------------------------------
# 16. COMPLETION
# ----------------------------------------------------------------

print("\n")
print("=" * 75)
print("PART (B) COMPLETED")
print("=" * 75)