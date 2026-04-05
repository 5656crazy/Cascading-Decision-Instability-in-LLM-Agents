# Cascading-Decision-Instability-in-LLM-Agents
This project investigates to what extent semantically equivalent prompt perturbations and minor threshold changes in data quality disrupt an LLM agent’s data cleaning and feature engineering decisions, and how this instability cascades into downstream model performance
# 🤖 Evaluating LLMOps: The Paradox of Intelligence in Automated Data Science

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_Workflow-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random_Forest-yellow)
![LLMs](https://img.shields.io/badge/LLMs-GPT--4o--mini%20%7C%20Claude--3.5--Sonnet-success)

## 📌 Executive Summary
Can Large Language Models (LLMs) fully automate a Data Science pipeline? This project conducts a rigorous empirical study (800 automated trials) comparing **GPT-4o-mini** and **Claude 3.5 Sonnet** in handling time-series data with varying missing rates (1% to 28%) and semantic prompt perturbations. 

The findings challenge the assumption that "smarter is always better." We uncovered the **"Paradox of Intelligence"**, where high-capability models over-analyze prompts, leading to severe feature hallucinations, while lightweight models exhibit high robustness. This repository contains the complete agentic workflow, evaluation metrics, and qualitative case studies.

---

## 🚀 Key Findings & Insights

### 1. The Paradox of Intelligence (Feature Hallucination)
High-capability models (Claude 3.5 Sonnet) are paradoxically more vulnerable to semantic noise (e.g., verbose business prompts) than lightweight models (GPT-4o-mini). While GPT maintained a near-perfect **Jaccard Similarity (~0.99)** across all prompt styles, Claude's attention was diluted by non-informative tokens, leading to a significant drop in **Rank-Biased Overlap (RBO)** and the loss of critical seasonal features.

### 2. The Deletion Bias (Decision Rigidity)
Regardless of the model's capability, LLM Agents exhibited a **100% preference for deleting missing data (`drop_rows`)** over imputation, even when the missing rate reached a critical 28%. Without hard-coded guardrails, this blind automation severely destroyed time-series continuity.

### 3. Non-linear Collapse & False Precision
* **Catastrophic Failure:** Crossing the 20% missing rate threshold triggered severe downstream RMSE spikes (up to 2400+ GBP), driven by LLM feature hallucinations.
* **The 10% Trap:** An anomalously low RMSE was observed at a 10% missing rate. This is a trap of **false precision**—the deletion of data inadvertently removed hard-to-predict peak outliers, artificially inflating apparent accuracy while destroying the model's ability to forecast critical real-world business peaks.

---

## 📊 Visualizing the Results

*(💡 Note: Upload your generated plots to an `images/` folder in this repository to render them below)*

### Head-to-Head Model Comparison (RMSE)
> GPT-4o-mini (Blue) vs. Claude 3.5 Sonnet (Orange). Notice the extreme variance and prediction collapse of Claude at high missing rates.

### Stability Metrics: Jaccard & RBO
> Measuring how feature lists and feature importance rankings shift under prompt perturbations.

---

## 🏗️ Architecture & Methodology

The pipeline is built using an Agentic Workflow representing a standard machine learning lifecycle:
1. **Cleaning Agent:** Decides how to handle missing values (`drop_rows`, `impute_mean`, etc.).
2. **Feature Agent:** Proposes time-series features (lags, rolling windows).
3. **Training & Evaluation:** Trains a `RandomForestRegressor` and extracts feature importance.

**Experimental Grid (800 Trials):**
* **Missing Rates:** 10 levels (0.01 to 0.28).
* **Prompt Styles:** 4 variations (`baseline`, `concise`, `verbose`, `reordered`).
* **Repetitions:** 10 trials per combination.
* **Models Tested:** `openai/gpt-4o-mini` vs `anthropic/claude-3.5-sonnet`.

---
cd [Your-Repo-Name]
pip install -r requirements.txt
