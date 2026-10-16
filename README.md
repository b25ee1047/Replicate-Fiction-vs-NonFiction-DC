# Genre Classification Replication and Paragraph Ablation Study

This project replicates the stylometric genre classification approach and conducts paragraph ablation analysis.

## Overview

The codebase analyzes text data from two corpora:
- Brown Corpus (324 documents across 10 genres)
- BNC Baby Corpus (55 documents: fiction and academic genres)

The classifier distinguishes fictional genres from non-fictional genres using L1-regularized Logistic Regression across 19 stylometric features.

---

## Paragraph Ablation Analysis

The module [paragraph_analysis.py](paragraph_analysis.py) examines the effect of removing paragraphs from each document.

### Removal Strategies Tested
1. **Random Paragraphs**: Removes $k$ randomly sampled paragraphs.
2. **First Paragraphs**: Removes the first $k$ paragraphs from the start of the document.
3. **Last Paragraphs**: Removes the last $k$ paragraphs from the end of the document.

### Removal Percentages
- Evaluated at: 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, and 90%.

### Evaluation Protocol
- 10 repeated 60/40 stratified train/test splits across 379 documents (Brown + BNC Baby).
- Evaluates four feature configurations:
  - All 19 Features
  - All Low-Level Features (11 features)
  - 6 Selected Features
  - 2 Selected Features

### Key Findings
- **High Stability (0% to 70% Removal)**: Model accuracy remains above 94% across all removal strategies. Stylometric ratios stay consistent throughout the text volume.
- **Degradation at Extreme Removal (80% to 90%)**: Accuracy drops when text volume becomes insufficient to represent stylistic distributions accurately.
- **Feature Set Robustness**: The full 19-feature set shows greater resilience to paragraph loss than the 2-feature subset at 90% reduction.

---

## Project Structure

- [main.py](main.py): Pipeline runner executing feature extraction, classification experiments, and figure generation.
- [paragraph_analysis.py](paragraph_analysis.py): Paragraph ablation experiments, caching, and plotting.
- [feature_selection.py](feature_selection.py): Feature definitions and two-stage RFECV logic.
- [experiments.py](experiments.py): Experimental setups and evaluation functions.
- [visualize.py](visualize.py): Visualization functions for distribution figures and scatter plots.
- [data_handling/](data_handling/): Corpus loaders and taggers for Brown and BNC Baby.

---

## Generated Plots

- [plots/paragraph_removal_accuracy.png](plots/paragraph_removal_accuracy.png): Four-panel plot of accuracy versus drop percentage across all feature sets.
- [plots/paragraph_removal_summary.png](plots/paragraph_removal_summary.png): Summary comparison for the 19-feature model with standard deviation error bars.
- [plots/figure1_scatter.png](plots/figure1_scatter.png): Genre cluster scatter plot.
- [plots/figure2_pos.png](plots/figure2_pos.png): Part-of-speech ratio distributions.
- [plots/figure3_punct.png](plots/figure3_punct.png): Punctuation ratio distributions.
- [plots/figure4_lowlevel.png](plots/figure4_lowlevel.png): Low-level feature distributions.

---

## JEFP Corpus External Evaluation (Optional)

This repository now supports optional evaluation on **JEFP** (Jena Expository vs Fictional Prose), while preserving the existing Brown/BNC workflow.

- JEFP categories preserved in outputs:
  - `canonical_fiction`
  - `noncanonical_fiction`
  - `non_fiction`
- Binary model label used for evaluation compatibility:
  - `fiction` = canonical + non-canonical fiction
  - `non_fiction` = non-fiction

### Important licensing note

This implementation **does not bundle or download JEFP text files**. You must provide a local JEFP text directory and metadata CSV that you are licensed to use.

### Expected local JEFP layout

A practical layout is:

```text
<JEFP_ROOT>/
  canonical/
    *.txt
  noncanonical/
    *.txt
  nonfiction/
    *.txt
```

The metadata CSV can point to files using a path-like column (for example `path`, `file_path`, `text_path`) or can be resolved by ID/filename-style columns (for example `id`, `doc_id`, `file_id`, `title`) against discovered `.txt` files under `JEFP_ROOT`.

### Run JEFP analysis directly

```bash
python run_jefp_analysis.py \
  --jefp-root /absolute/path/to/jefp/texts \
  --jefp-metadata /absolute/path/to/jefp_metadata.csv
```

Useful options:

- `--inspect-only` to preview metadata/path resolution without feature extraction
- `--scratch-dir` and `--plots-dir` to override output directories
- `--six-features` / `--two-features` to provide comma-separated selected feature sets
- `--no-derive-feature-sets` to skip RFECV-derived 6/2 feature extraction

### Run JEFP stage via `main.py`

`main.py` remains safe by default. JEFP runs only when explicitly enabled:

```bash
ENABLE_JEFP=1 \
JEFP_ROOT=/absolute/path/to/jefp/texts \
JEFP_METADATA=/absolute/path/to/jefp_metadata.csv \
python main.py
```

If files are missing, the script prints a concise skip message and continues without failing the default Brown/BNC pipeline.

### JEFP outputs

- Tables/CSVs: `scratch/jefp/`
  - `metrics_by_feature_set.csv`
  - `document_predictions.csv`
  - `category_distribution.csv`
  - `confusion_matrix_*.csv`
  - `canonical_noncanonical_breakdown.csv` (when available)
- Plots: `plots/jefp/`
  - JEFP class/category distribution
  - Confusion matrix heatmaps by feature set
  - Accuracy/F1 comparison across feature sets
  - Prediction-confidence distributions by true category and predicted class
  - Selected-feature distributions across JEFP categories
  - Canonical vs non-canonical fiction performance comparison

### Interpretation cautions

- JEFP differs from Brown/BNC in domain and document length, so expect domain shift effects.
- Long documents can stabilize stylometric estimates differently from shorter corpora.
- Always interpret JEFP results alongside corpus licensing constraints and metadata quality.
