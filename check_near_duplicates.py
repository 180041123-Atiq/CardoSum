import os
import pandas as pd
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

TRAIN_FILE = "train.csv"
VAL_FILE = "val.csv"
TEST_FILE = "test.csv"

TEXT_COL = "Original_Report"

# Experiment settings
N_GRAMS = [20, 25, 30]
THRESHOLDS = [0.98, 0.97, 0.96]

NUM_PERM = 128

RESULTS_DIR = "results"
SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "near_duplicate_summary.csv"
)


# ============================================================
# Character n-gram generation
# ============================================================

def char_ngrams(text, n):
    """
    Generate character-level n-grams.
    """
    text = str(text).lower().strip()

    if len(text) < n:
        return {text}

    return {
        text[i:i+n]
        for i in range(len(text) - n + 1)
    }


# ============================================================
# Create MinHash
# ============================================================

def create_minhash(text, n):
    """
    Create a MinHash signature from character n-grams.
    """
    m = MinHash(num_perm=NUM_PERM)

    for gram in char_ngrams(text, n):
        m.update(gram.encode("utf-8"))

    return m


# ============================================================
# Build MinHash index
# ============================================================

def build_lsh(data, prefix, n_gram, threshold):
    """
    Build MinHash LSH index for a dataset.
    """

    lsh = MinHashLSH(
        threshold=threshold,
        num_perm=NUM_PERM
    )

    minhashes = {}

    for idx, text in tqdm(
        data[TEXT_COL].items(),
        total=len(data),
        desc=f"Indexing {prefix}"
    ):

        mh = create_minhash(text, n_gram)

        key = f"{prefix}_{idx}"

        lsh.insert(key, mh)
        minhashes[key] = mh

    return lsh, minhashes


# ============================================================
# Compare source dataset against reference dataset
# ============================================================

def calculate_match_percentage(
    source_data,
    reference_lsh,
    reference_minhashes,
    source_prefix,
    reference_prefix,
    n_gram,
    threshold
):
    """
    Calculate the percentage of source samples that have
    at least one reference sample with Jaccard similarity
    >= threshold.

    Returns:
        match_percentage
        number_of_matches
    """

    matched_count = 0

    for idx, text in tqdm(
        source_data[TEXT_COL].items(),
        total=len(source_data),
        desc=f"{source_prefix} vs {reference_prefix}"
    ):

        source_mh = create_minhash(text, n_gram)

        # LSH returns candidates likely to satisfy the threshold
        candidates = reference_lsh.query(source_mh)

        found_match = False

        for candidate in candidates:

            similarity = source_mh.jaccard(
                reference_minhashes[candidate]
            )

            if similarity >= threshold:
                found_match = True
                break

        if found_match:
            matched_count += 1

    total_samples = len(source_data)

    match_percentage = (
        100 * matched_count / total_samples
        if total_samples > 0
        else 0
    )

    return match_percentage, matched_count


# ============================================================
# Run one experiment
# ============================================================

def run_experiment(
    train,
    val,
    test,
    n_gram,
    threshold
):

    print("\n" + "=" * 80)
    print(f"N-GRAM = {n_gram}, THRESHOLD = {threshold}")
    print("=" * 80)

    results = []

    # --------------------------------------------------------
    # Build TRAIN index
    # --------------------------------------------------------

    print("\nBuilding TRAIN index...")

    train_lsh, train_minhashes = build_lsh(
        train,
        "train",
        n_gram,
        threshold
    )

    # --------------------------------------------------------
    # TEST vs TRAIN
    # --------------------------------------------------------

    print("\nChecking TEST vs TRAIN...")

    test_train_percentage, test_train_count = (
        calculate_match_percentage(
            test,
            train_lsh,
            train_minhashes,
            "test",
            "train",
            n_gram,
            threshold
        )
    )

    print(
        f"TEST vs TRAIN: "
        f"{test_train_count}/{len(test)} "
        f"({test_train_percentage:.2f}%)"
    )

    results.append({
        "setup": "testVStrain",
        "n_gram": n_gram,
        "threshold": threshold,
        "Percentage": test_train_percentage
    })

    # --------------------------------------------------------
    # VAL vs TRAIN
    # --------------------------------------------------------

    print("\nChecking VAL vs TRAIN...")

    val_train_percentage, val_train_count = (
        calculate_match_percentage(
            val,
            train_lsh,
            train_minhashes,
            "val",
            "train",
            n_gram,
            threshold
        )
    )

    print(
        f"VAL vs TRAIN: "
        f"{val_train_count}/{len(val)} "
        f"({val_train_percentage:.2f}%)"
    )

    results.append({
        "setup": "valVStrain",
        "n_gram": n_gram,
        "threshold": threshold,
        "Percentage": val_train_percentage
    })

    # --------------------------------------------------------
    # Build VALIDATION index
    # --------------------------------------------------------

    print("\nBuilding VALIDATION index...")

    val_lsh, val_minhashes = build_lsh(
        val,
        "val",
        n_gram,
        threshold
    )

    # --------------------------------------------------------
    # TEST vs VALIDATION
    # --------------------------------------------------------

    print("\nChecking TEST vs VALIDATION...")

    test_val_percentage, test_val_count = (
        calculate_match_percentage(
            test,
            val_lsh,
            val_minhashes,
            "test",
            "val",
            n_gram,
            threshold
        )
    )

    print(
        f"TEST vs VALIDATION: "
        f"{test_val_count}/{len(test)} "
        f"({test_val_percentage:.2f}%)"
    )

    results.append({
        "setup": "testVSval",
        "n_gram": n_gram,
        "threshold": threshold,
        "Percentage": test_val_percentage
    })

    return results


# ============================================================
# Main
# ============================================================

def check_near_dup():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    train = pd.read_csv(TRAIN_FILE)
    val = pd.read_csv(VAL_FILE)
    test = pd.read_csv(TEST_FILE)

    print("Dataset sizes:")
    print(f"Train: {len(train)}")
    print(f"Validation: {len(val)}")
    print(f"Test: {len(test)}")

    # --------------------------------------------------------
    # Run all experiments
    # --------------------------------------------------------

    all_results = []

    for n_gram in N_GRAMS:

        for threshold in THRESHOLDS:

            experiment_results = run_experiment(
                train=train,
                val=val,
                test=test,
                n_gram=n_gram,
                threshold=threshold
            )

            all_results.extend(experiment_results)

    # --------------------------------------------------------
    # Create summary CSV
    # --------------------------------------------------------

    summary_df = pd.DataFrame(all_results)

    summary_df = summary_df[
        [
            "setup",
            "n_gram",
            "threshold",
            "Percentage"
        ]
    ]

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Print final summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        summary_df.to_string(index=False)
    )

    print(
        f"\nSummary saved to: {SUMMARY_FILE}"
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    check_near_dup()