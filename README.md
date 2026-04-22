# Warranty Data Analysis & Root-Cause Tool

A Jupyter-Notebook-based pipeline that ingests dealer warranty repair-history
data (English `.xlsx`) and produces:

1. Variable-role analysis (input vs. output) with rationale
2. Significance-tested trend statistics at **95% confidence (alpha = 0.05)**
3. Regression modeling of defect drivers (logistic / Poisson / OLS / multinomial)
   with odds ratios / coefficients and 95% CIs
4. LLM-based symptom binning of technician comments (13 standard automotive
   system categories)
5. DTC extraction + LLM judgement on whether the DTC matches the customer
   complaint and the technician comment
6. Web-search verification of DTC real-world set conditions
7. Web-search verification of the repair method used
8. Specific failing system / part determination
9. Optional correlation with connected-vehicle (CV) data dropped into
   `data/connected/`
10. A final English briefing (inline markdown + Excel `Summary` sheet +
    optional PDF)

## Quick start

```bash
# 1. Install
pip install -e .

# 2. Provide your key
cp .env.example .env    # then edit .env

# 3. Drop your warranty Excel into data/input/
# 4. Launch the notebook
jupyter lab notebooks/warranty_analysis.ipynb
```

Run cells in order. Each cell shows a preview of intermediate results for
review before the next step is run. The original Excel is never modified in
place; results are written to `data/output/<original>_analyzed_<timestamp>.xlsx`
with 11 new columns appended and a `Summary` sheet.

## Project layout

```
src/warranty/          Python modules
config/                settings.yaml, taxonomy.yaml, column_aliases.yaml
notebooks/             Main notebook
data/input/            Drop warranty .xlsx here
data/connected/        Drop connected-vehicle CSV/Parquet here (optional)
data/output/           Analyzed Excel + charts + briefing
docs/FAILURE_LOG.md    Persistent log of failure causes & lessons
tests/                 pytest suite
scripts/               Synthetic data generator for smoke testing
```

## Conventions

All code, comments, LLM prompts, notebook markdown, and briefings are in
English. Every module has Google-style docstrings and inline comments
explaining non-trivial logic. Every `except` block logs to
`docs/FAILURE_LOG.md` via `failure_log.record()`.
