import os
import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.stats.multitest import multipletests

from rouge_score import rouge_scorer
from bert_score import score as bert_score
from datasets import load_dataset
from tqdm import tqdm

from csr_reward_fn import calc_piecewise_readability_reward, calc_piecewise_conciseness_reward, calc_similarity_reward
# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = "test.csv"

SFT_FOLDER = "gen_outputs_sft_43"
HVO_FOLDER = "gen_outputs_piecewise_hvo_43"

OUTPUT_EXCEL = "statistical_analysis.xlsx"

N_SAMPLES = 1500

# Confidence level
ALPHA = 0.05


# ============================================================
# THREE-SEED RESULTS
# ============================================================

seed_results = {
    "ROUGE-L": {
        "SFT": [0.3298, 0.32796, 0.3305],
        "HVO": [0.3123, 0.3112, 0.3085],
    },

    "BERTScore": {
        "SFT": [0.8954, 0.8952, 0.8952],
        "HVO": [0.8873, 0.8872, 0.8857],
    },

    "Readability": {
        "SFT": [0.4792, 0.4786, 0.481],
        "HVO": [0.7018, 0.6992, 0.7074],
    },

    "Conciseness": {
        "SFT": [0.7054, 0.7031, 0.7034],
        "HVO": [0.7499, 0.7552, 0.7574],
    },

    "Similarity": {
        "SFT": [0.8221, 0.8284, 0.8278],
        "HVO": [0.8146, 0.8159, 0.8141],
    }
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def mean_sd_ci(values, alpha=0.05):
    """
    Calculate:
        mean
        sample SD
        95% CI of the mean

    CI is based on the t-distribution.
    """

    values = np.asarray(values, dtype=float)

    n = len(values)

    mean = np.mean(values)

    sd = np.std(values, ddof=1)

    se = sd / np.sqrt(n)

    t_critical = stats.t.ppf(
        1 - alpha / 2,
        df=n - 1
    )

    margin = t_critical * se

    lower = mean - margin
    upper = mean + margin

    return mean, sd, lower, upper


def paired_mean_ci(a, b, alpha=0.05):
    """
    Calculate 95% CI for the paired mean difference.

    Difference = a - b
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    d = a - b

    n = len(d)

    mean_diff = np.mean(d)

    sd_diff = np.std(d, ddof=1)

    se = sd_diff / np.sqrt(n)

    t_critical = stats.t.ppf(
        1 - alpha / 2,
        df=n - 1
    )

    margin = t_critical * se

    lower = mean_diff - margin
    upper = mean_diff + margin

    return mean_diff, sd_diff, lower, upper


# ============================================================
# PART 1:
# THREE-SEED ANALYSIS
# ============================================================

seed_summary = []

for metric, methods in seed_results.items():

    sft = methods["SFT"]
    hvo = methods["HVO"]

    sft_mean, sft_sd, sft_ci_low, sft_ci_high = \
        mean_sd_ci(sft)

    hvo_mean, hvo_sd, hvo_ci_low, hvo_ci_high = \
        mean_sd_ci(hvo)

    # Paired test across the three seeds
    t_stat, p_value = stats.ttest_rel(
        sft,
        hvo
    )

    seed_summary.append({

        "Metric": metric,

        "SFT Mean": sft_mean,
        "SFT SD": sft_sd,
        "SFT 95% CI Lower": sft_ci_low,
        "SFT 95% CI Upper": sft_ci_high,

        "HVO Mean": hvo_mean,
        "HVO SD": hvo_sd,
        "HVO 95% CI Lower": hvo_ci_low,
        "HVO 95% CI Upper": hvo_ci_high,

        "Mean Difference": sft_mean - hvo_mean,

        "Paired t-statistic": t_stat,
        "Paired p-value": p_value
    })


seed_summary_df = pd.DataFrame(seed_summary)


# ============================================================
# PART 2:
# LOAD THE 1500 TEST REPORTS
# ============================================================

# df = pd.read_csv(CSV_FILE)

# if "Original_Report" not in df.columns:
#     raise ValueError(
#         "CSV must contain an 'Original_Report' column."
#     )

# if len(df) < N_SAMPLES:
#     raise ValueError(
#         f"CSV contains only {len(df)} samples."
#     )

# # references = (
# #     df["Original_Report"]
# #     .iloc[:N_SAMPLES]
# #     .fillna("")
# #     .astype(str)
# #     .tolist()
# # )

prompts,completions,references = [],[],[]
test_dataset = load_dataset("csv",data_files="test.csv",)['train']

# sft_output_dir = 'gen_outputs_piecewise_hvo_44'
    
for ii,sample in enumerate(tqdm(test_dataset, total=len(test_dataset), desc='Converting samples for reward calc')):
    prompts.append(sample['Original_Report'])
    references.append(sample['GPT_Summary'])
    # completion = ''
    # with open(f'{output_dir}/{ii}.txt','r') as f:
    #     completion = f.read()
    
    # completions.append(completion)


# ============================================================
# LOAD GENERATED SUMMARIES
# ============================================================

def load_summaries(folder, n_samples):

    summaries = []

    for i in range(n_samples):

        filename = os.path.join(
            folder,
            f"{i}.txt"
        )

        if not os.path.exists(filename):

            raise FileNotFoundError(
                f"Missing summary file: {filename}"
            )

        with open(
            filename,
            "r",
            encoding="utf-8"
        ) as f:

            text = f.read().strip()

        summaries.append(text)

    return summaries


print("\nLoading SFT summaries...")

sft_summaries = load_summaries(
    SFT_FOLDER,
    N_SAMPLES
)

print("Loading HVO summaries...")

hvo_summaries = load_summaries(
    HVO_FOLDER,
    N_SAMPLES
)


assert len(references) == len(sft_summaries)
assert len(references) == len(hvo_summaries)


# ============================================================
# PART 3:
# ROUGE-L PER SAMPLE
# ============================================================

print("\nCalculating per-sample ROUGE-L...")

rouge_scorer_obj = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True
)


def calculate_rouge_l(references, predictions):

    scores = []

    for ref, pred in zip(
        references,
        predictions
    ):

        result = rouge_scorer_obj.score(
            ref,
            pred
        )

        scores.append(
            result["rougeL"].fmeasure
        )

    return np.array(scores)


sft_rouge = calculate_rouge_l(
    references,
    sft_summaries
)

hvo_rouge = calculate_rouge_l(
    references,
    hvo_summaries
)


print(
    "SFT ROUGE-L:",
    np.mean(sft_rouge)
)

print(
    "HVO ROUGE-L:",
    np.mean(hvo_rouge)
)


# ============================================================
# PART 4:
# BERTSCORE PER SAMPLE
# ============================================================

print("\nCalculating BERTScore...")

# Change lang if your reports are not English.
P, R, F1 = bert_score(
    sft_summaries,
    references,
    lang="en",
    verbose=True
)

sft_bertscore = F1.cpu().numpy()


P, R, F1 = bert_score(
    hvo_summaries,
    references,
    lang="en",
    verbose=True
)

hvo_bertscore = F1.cpu().numpy()


# ============================================================
# PART 5:
# ADD YOUR OTHER PER-SAMPLE METRICS
# ============================================================

"""
IMPORTANT:

You need to calculate the following metrics using EXACTLY
the same formulas that you used to produce your reported
Readability, Conciseness, and Similarity values.

The statistical code below expects:

    sft_readability
    hvo_readability

    sft_conciseness
    hvo_conciseness

    sft_similarity
    hvo_similarity

each to contain 1500 values.

For now, these are placeholders.

Replace these sections with your actual metric calculations.
"""


# For Piecewise Readability and Conciseness
sft_readability = calc_piecewise_readability_reward(sft_summaries)
sft_conciseness = calc_piecewise_conciseness_reward(prompts,sft_summaries)
sft_similarity = calc_similarity_reward(sft_summaries,references)

hvo_readability = calc_piecewise_readability_reward(hvo_summaries)
hvo_conciseness = calc_piecewise_conciseness_reward(prompts,hvo_summaries)
hvo_similarity = calc_similarity_reward(hvo_summaries,references)

# ------------------------------------------------------------
# Example:
#
# sft_readability = calculate_readability(sft_summaries)
# hvo_readability = calculate_readability(hvo_summaries)
#
# sft_conciseness = calculate_conciseness(sft_summaries)
# hvo_conciseness = calculate_conciseness(hvo_summaries)
#
# sft_similarity = calculate_similarity(
#     references,
#     sft_summaries
# )
#
# hvo_similarity = calculate_similarity(
#     references,
#     hvo_summaries
# )
# ------------------------------------------------------------


# TEMPORARY:
# Replace these with your actual arrays.

sft_readability = np.array(sft_readability)
hvo_readability = np.array(hvo_readability)

sft_conciseness = np.array(sft_conciseness)
hvo_conciseness = np.array(hvo_conciseness)

sft_similarity = np.array(sft_similarity)
hvo_similarity = np.array(hvo_similarity)


# ============================================================
# PART 6:
# COLLECT ALL FIVE METRICS
# ============================================================

per_sample_metrics = {

    "ROUGE-L": (
        sft_rouge,
        hvo_rouge
    ),

    "BERTScore": (
        sft_bertscore,
        hvo_bertscore
    ),

    "Readability": (
        sft_readability,
        hvo_readability
    ),

    "Conciseness": (
        sft_conciseness,
        hvo_conciseness
    ),

    "Similarity": (
        sft_similarity,
        hvo_similarity
    )
}


# ============================================================
# PART 7:
# PAIRED STATISTICAL TEST
# ============================================================

test_results = []

raw_p_values = []


for metric, (sft, hvo) in per_sample_metrics.items():

    sft = np.asarray(sft, dtype=float)
    hvo = np.asarray(hvo, dtype=float)

    if len(sft) != N_SAMPLES:
        raise ValueError(
            f"{metric}: SFT has {len(sft)} samples."
        )

    if len(hvo) != N_SAMPLES:
        raise ValueError(
            f"{metric}: HVO has {len(hvo)} samples."
        )

    # --------------------------------------------------------
    # Mean scores
    # --------------------------------------------------------

    sft_mean = np.mean(sft)
    hvo_mean = np.mean(hvo)

    # --------------------------------------------------------
    # Paired difference
    # --------------------------------------------------------

    mean_diff, sd_diff, ci_low, ci_high = \
        paired_mean_ci(
            sft,
            hvo
        )

    # --------------------------------------------------------
    # Paired t-test
    # --------------------------------------------------------

    t_stat, t_p = stats.ttest_rel(
        sft,
        hvo
    )

    # --------------------------------------------------------
    # Wilcoxon signed-rank test
    # --------------------------------------------------------

    try:

        w_stat, w_p = stats.wilcoxon(
            sft,
            hvo,
            alternative="two-sided"
        )

    except ValueError:

        w_stat = np.nan
        w_p = np.nan

    raw_p_values.append(w_p)

    test_results.append({

        "Metric": metric,

        "SFT Mean": sft_mean,
        "HVO Mean": hvo_mean,

        "Mean Difference (SFT-HVO)": mean_diff,

        "Difference SD": sd_diff,

        "95% CI Lower": ci_low,
        "95% CI Upper": ci_high,

        "Paired t-statistic": t_stat,
        "Paired t-test p": t_p,

        "Wilcoxon statistic": w_stat,
        "Wilcoxon p": w_p
    })


test_results_df = pd.DataFrame(
    test_results
)


# ============================================================
# PART 8:
# HOLM-BONFERRONI CORRECTION
# ============================================================

reject, adjusted_p, _, _ = multipletests(
    raw_p_values,
    alpha=ALPHA,
    method="holm"
)

test_results_df[
    "Holm-adjusted p"
] = adjusted_p

test_results_df[
    "Significant after Holm"
] = reject


# ============================================================
# PART 9:
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 90)
print("THREE-SEED RESULTS")
print("=" * 90)

print(
    seed_summary_df.to_string(
        index=False
    )
)


print("\n")
print("=" * 90)
print("1500-SAMPLE PAIRED TEST")
print("=" * 90)

print(
    test_results_df.to_string(
        index=False
    )
)


# ============================================================
# PART 10:
# SAVE TO EXCEL
# ============================================================

with pd.ExcelWriter(
    OUTPUT_EXCEL,
    engine="openpyxl"
) as writer:

    seed_summary_df.to_excel(
        writer,
        sheet_name="Three_Seed_Results",
        index=False
    )

    test_results_df.to_excel(
        writer,
        sheet_name="1500_Sample_Tests",
        index=False
    )


print("\n")
print("=" * 90)
print(
    f"Results saved to: {OUTPUT_EXCEL}"
)
print("=" * 90)