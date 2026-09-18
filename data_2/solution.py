import pandas as pd
import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# QUESTION 2(a)
# MECHANICAL DEFINITION OF "SIMILAR"
#
# COMPETING CHOICES:
# 1. WORD-LEVEL TF-IDF
# 2. CHARACTER-LEVEL TF-IDF
#
# TRUSTED LABEL SOURCE:
# labelled_pairs.csv
# ============================================================


DATA_PATH = "."


print("=" * 75)
print("QUESTION 2(a) - MECHANICAL SIMILARITY")
print("=" * 75)


# ============================================================
# 1. LOAD THE COMPLETE CORPUS
# ============================================================

notice_files = glob.glob(
    os.path.join(DATA_PATH, "notices", "*.csv")
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

print("\nNotice columns:")
print(notices.columns.tolist())


# ============================================================
# 2. TEXT PREPROCESSING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    # Remove HTML
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Keep letters and numbers
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Normalize spaces
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

# Give title equal combined importance with body
notices["text"] = (
    notices["title_clean"]
    + " "
    + notices["body_clean"]
)

print("\nText preprocessing completed.")


# ============================================================
# 3. INSPECT NUMBERS / IDENTIFIERS
# ============================================================

print("\nHandling of special fields:")
print("- Monetary amounts, dates and reference numbers remain")
print("  in the original notice metadata.")
print("- Text similarity removes punctuation and normalizes text.")
print("- Portal boilerplate is reduced using TF-IDF stop-word filtering.")
print("- Word and character representations are compared empirically.")


# ============================================================
# 4. LOAD TRUSTWORTHY LABELS
# ============================================================

pairs = pd.read_csv(
    os.path.join(DATA_PATH, "labelled_pairs.csv")
)

print("\nNumber of labelled pairs:", len(pairs))

print("\nLabel distribution:")
print(pairs["label"].value_counts())


# ============================================================
# 5. NOTICE ID INDEX
# ============================================================

notice_index = {
    str(notices.iloc[i]["notice_id"]): i
    for i in range(len(notices))
}


# ============================================================
# 6. COMPETING CHOICE 1
# WORD-LEVEL TF-IDF
# ============================================================

print("\n")
print("=" * 75)
print("CHOICE 1 - WORD-LEVEL TF-IDF")
print("=" * 75)

word_vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

word_matrix = word_vectorizer.fit_transform(
    notices["text"]
)

print(
    "Word TF-IDF matrix:",
    word_matrix.shape
)


# ============================================================
# 7. COMPETING CHOICE 2
# CHARACTER-LEVEL TF-IDF
# ============================================================

print("\n")
print("=" * 75)
print("CHOICE 2 - CHARACTER-LEVEL TF-IDF")
print("=" * 75)

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=5000,
    min_df=2
)

char_matrix = char_vectorizer.fit_transform(
    notices["text"]
)

print(
    "Character TF-IDF matrix:",
    char_matrix.shape
)


# ============================================================
# 8. FUNCTION TO CALCULATE PAIR SCORES
# ============================================================

def calculate_scores(matrix):

    scores = []

    for _, row in pairs.iterrows():

        id_a = str(row["notice_id_a"])
        id_b = str(row["notice_id_b"])

        if id_a in notice_index and id_b in notice_index:

            index_a = notice_index[id_a]
            index_b = notice_index[id_b]

            score = cosine_similarity(
                matrix[index_a],
                matrix[index_b]
            )[0][0]

            scores.append(score)

        else:

            scores.append(np.nan)

    return np.array(scores)


# ============================================================
# 9. CALCULATE BOTH SETS OF SCORES
# ============================================================

word_scores = calculate_scores(
    word_matrix
)

char_scores = calculate_scores(
    char_matrix
)

pairs["word_similarity"] = word_scores
pairs["char_similarity"] = char_scores

pairs_eval = pairs.dropna(
    subset=[
        "word_similarity",
        "char_similarity"
    ]
).copy()

pairs_eval["actual"] = (
    pairs_eval["label"]
    .str.lower()
    .eq("same")
    .astype(int)
)

print("\nPairs successfully evaluated:", len(pairs_eval))


# ============================================================
# 10. COMPARE A SAME PAIR AND A DIFFERENT PAIR
# ============================================================

same_pair = pairs_eval[
    pairs_eval["label"].str.lower() == "same"
].iloc[0]

different_pair = pairs_eval[
    pairs_eval["label"].str.lower() == "different"
].iloc[0]


print("\n")
print("=" * 75)
print("PAIR-LEVEL COMPARISON")
print("=" * 75)

print("\nSAME PAIR:")
print(
    "Notice A:",
    same_pair["notice_id_a"]
)

print(
    "Notice B:",
    same_pair["notice_id_b"]
)

print(
    "Word TF-IDF similarity:",
    f"{same_pair['word_similarity']:.6f}"
)

print(
    "Character TF-IDF similarity:",
    f"{same_pair['char_similarity']:.6f}"
)


print("\nDIFFERENT PAIR:")
print(
    "Notice A:",
    different_pair["notice_id_a"]
)

print(
    "Notice B:",
    different_pair["notice_id_b"]
)

print(
    "Word TF-IDF similarity:",
    f"{different_pair['word_similarity']:.6f}"
)

print(
    "Character TF-IDF similarity:",
    f"{different_pair['char_similarity']:.6f}"
)


# ============================================================
# 11. AVERAGE SEPARATION
# ============================================================

print("\n")
print("=" * 75)
print("AVERAGE SIMILARITY BY LABEL")
print("=" * 75)

print("\nWORD TF-IDF:")

print(
    pairs_eval
    .groupby("label")["word_similarity"]
    .mean()
)

print("\nCHARACTER TF-IDF:")

print(
    pairs_eval
    .groupby("label")["char_similarity"]
    .mean()
)


# ============================================================
# 12. THRESHOLD TESTING FUNCTION
# ============================================================

def evaluate_representation(
    scores,
    actual
):

    results = []

    for threshold in np.arange(
        0.10,
        0.96,
        0.05
    ):

        predicted = (
            scores >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            actual,
            predicted
        )

        precision = precision_score(
            actual,
            predicted,
            zero_division=0
        )

        recall = recall_score(
            actual,
            predicted,
            zero_division=0
        )

        f1 = f1_score(
            actual,
            predicted,
            zero_division=0
        )

        results.append([
            threshold,
            accuracy,
            precision,
            recall,
            f1
        ])

    return pd.DataFrame(
        results,
        columns=[
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]
    )


# ============================================================
# 13. EVALUATE WORD REPRESENTATION
# ============================================================

word_results = evaluate_representation(
    pairs_eval["word_similarity"].values,
    pairs_eval["actual"].values
)

best_word = word_results.loc[
    word_results["f1"].idxmax()
]


# ============================================================
# 14. EVALUATE CHARACTER REPRESENTATION
# ============================================================

char_results = evaluate_representation(
    pairs_eval["char_similarity"].values,
    pairs_eval["actual"].values
)

best_char = char_results.loc[
    char_results["f1"].idxmax()
]


# ============================================================
# 15. DISPLAY COMPETING RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("COMPETING REPRESENTATION RESULTS")
print("=" * 75)

print("\nWORD TF-IDF")
print(
    "Best threshold:",
    best_word["threshold"]
)

print(
    "Accuracy:",
    best_word["accuracy"]
)

print(
    "Precision:",
    best_word["precision"]
)

print(
    "Recall:",
    best_word["recall"]
)

print(
    "F1:",
    best_word["f1"]
)


print("\nCHARACTER TF-IDF")
print(
    "Best threshold:",
    best_char["threshold"]
)

print(
    "Accuracy:",
    best_char["accuracy"]
)

print(
    "Precision:",
    best_char["precision"]
)

print(
    "Recall:",
    best_char["recall"]
)

print(
    "F1:",
    best_char["f1"]
)


# ============================================================
# 16. CONFUSION MATRICES
# ============================================================

word_predicted = (
    pairs_eval["word_similarity"]
    >= best_word["threshold"]
).astype(int)

char_predicted = (
    pairs_eval["char_similarity"]
    >= best_char["threshold"]
).astype(int)


word_cm = confusion_matrix(
    pairs_eval["actual"],
    word_predicted
)

char_cm = confusion_matrix(
    pairs_eval["actual"],
    char_predicted
)


print("\n")
print("=" * 75)
print("WORD TF-IDF CONFUSION MATRIX")
print("=" * 75)

print(word_cm)


print("\n")
print("=" * 75)
print("CHARACTER TF-IDF CONFUSION MATRIX")
print("=" * 75)

print(char_cm)


# ============================================================
# 17. PLOT SCORE SEPARATION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    pairs_eval[
        pairs_eval["label"] == "same"
    ]["word_similarity"],
    bins=30,
    alpha=0.6,
    label="Word TF-IDF - Same"
)

plt.hist(
    pairs_eval[
        pairs_eval["label"] == "different"
    ]["word_similarity"],
    bins=30,
    alpha=0.6,
    label="Word TF-IDF - Different"
)

plt.axvline(
    best_word["threshold"],
    linestyle="--",
    label="Word threshold"
)

plt.xlabel(
    "Cosine Similarity"
)

plt.ylabel(
    "Number of Pairs"
)

plt.title(
    "Word TF-IDF Similarity Separation"
)

plt.legend()
plt.grid(True)

plt.savefig(
    "q2a_word_similarity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 18. CHARACTER SCORE SEPARATION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    pairs_eval[
        pairs_eval["label"] == "same"
    ]["char_similarity"],
    bins=30,
    alpha=0.6,
    label="Character TF-IDF - Same"
)

plt.hist(
    pairs_eval[
        pairs_eval["label"] == "different"
    ]["char_similarity"],
    bins=30,
    alpha=0.6,
    label="Character TF-IDF - Different"
)

plt.axvline(
    best_char["threshold"],
    linestyle="--",
    label="Character threshold"
)

plt.xlabel(
    "Cosine Similarity"
)

plt.ylabel(
    "Number of Pairs"
)

plt.title(
    "Character TF-IDF Similarity Separation"
)

plt.legend()
plt.grid(True)

plt.savefig(
    "q2a_character_similarity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 19. FINAL ADOPTED METHOD
# ============================================================

if best_word["f1"] >= best_char["f1"]:

    adopted_method = "Word-level TF-IDF"

    adopted_threshold = best_word["threshold"]

    adopted_accuracy = best_word["accuracy"]

    adopted_precision = best_word["precision"]

    adopted_recall = best_word["recall"]

    adopted_f1 = best_word["f1"]

else:

    adopted_method = "Character-level TF-IDF"

    adopted_threshold = best_char["threshold"]

    adopted_accuracy = best_char["accuracy"]

    adopted_precision = best_char["precision"]

    adopted_recall = best_char["recall"]

    adopted_f1 = best_char["f1"]


print("\n")
print("=" * 75)
print("ADOPTED SIMILARITY DEFINITION")
print("=" * 75)

print(
    "Method:",
    adopted_method
)

print(
    "Similarity score: Cosine similarity"
)

print(
    "Threshold:",
    adopted_threshold
)

print(
    "Accuracy:",
    adopted_accuracy
)

print(
    "Precision:",
    adopted_precision
)

print(
    "Recall:",
    adopted_recall
)

print(
    "F1 Score:",
    adopted_f1
)


# ============================================================
# 20. FINAL INTERPRETATION
# ============================================================

print("\n")
print("=" * 75)
print("INTERPRETATION")
print("=" * 75)

print(
    "A pair is mechanically considered similar when its"
)

print(
    "cosine similarity is greater than or equal to the"
)

print(
    "selected threshold."
)

print(
    "\nTwo competing representations were evaluated:"
)

print(
    "1. Word-level TF-IDF using word unigrams and bigrams."
)

print(
    "2. Character-level TF-IDF using character n-grams."
)

print(
    "\nThe final representation was selected from the"
)

print(
    "labelled-pair measurements rather than from a"
)

print(
    "tutorial or arbitrary parameter."
)


# ============================================================
# 21. COMPLETED
# ============================================================

print("\n")
print("=" * 75)
print("PART (A) COMPLETED")
print("=" * 75)