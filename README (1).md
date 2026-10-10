# Financial Services: Consumer Complaint Intelligence

E1X Academy Data Science Capstone Project

**Group members:** Group1

## 1. Project overview

Financial services organisations receive large numbers of consumer complaints. Today, agents read each complaint and route it to a resolution team by judgment, which is slow and inconsistent between agents.

**Problem we chose:** predict the **financial product category** of a complaint (for example, "Debt collection" or "Credit reporting") from the consumer's own narrative, so that intake staff get a suggested route and a confidence score. The system is meant to **support staff, not replace them**.

**Alternative we considered:** priority scoring, which uses keyword signals (fraud, legal threats, financial loss and so on) to rank complaints by urgency. We built it as an exploratory comparison and did not choose it. Keyword rules can miss synonyms, and its labels come from our own rules rather than from real outcomes.

## 2. Dataset

- **Source:** US Consumer Financial Protection Bureau (CFPB) complaint database, mirrored on Kaggle: <https://www.kaggle.com/datasets/iuriivoloshyn/cfpb-consumer-complaint-database>
- **File used by the notebook:** `CFPB_Consumer_Complaints_2024 (1).csv`
- **Size of the file used for the saved results:** 500 complaints and 18 columns. Only 200 of them (40%) contain a consumer narrative, because the CFPB publishes narratives only when consumers consent. All text modelling is done on those 200 rows.
- The data file is **not included** in this submission. Download it from the Kaggle link above.

## 3. Repository contents

| File | Description |
|---|---|
| `Financial_services_capstone_project_FIXED_1_.ipynb` | The full analysis: EDA, baseline, alternative approach, model, threshold analysis, Data Critique, Decision Log and Business Recommendation |
| Presentation deck (`.pptx`) | 10-slide summary of the project |
| `README.md` | This file |

## 4. Requirements

- Python 3.9 or newer
- Packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## 5. How to run

1. Download the CSV from the Kaggle link and place it in the **same folder** as the notebook.
2. Check that the file name matches the one in the notebook's third cell (`pd.read_csv(...)`). If your file has a different name, edit that line.
3. Start Jupyter and open the notebook:
   ```bash
   jupyter notebook Financial_services_capstone_project_FIXED_1_.ipynb
   ```
4. Choose **Kernel → Restart & Run All**.

**Important:** run the notebook from top to bottom. The same variable names (`model`, `X_train`, `y_train`, `y_pred`) are used first for the product classifier (cells 70 to 77) and then again for the priority-scoring model (cells 78 to 85). Running cells out of order will mix the two sets of results. The random seed is fixed (`random_state=42`) and the test set is 20% of the data, so results are reproducible for a given input file.

## 6. How the notebook is organised

| Section | What it covers |
|---|---|
| Exploratory Data Analysis | Dataset shape, missing values, removal of rows without narratives |
| Text characteristics | Narrative length, word counts, average word length, digits and uppercase letters |
| Complaint distribution | Share of complaints per product, showing the class imbalance |
| Time patterns | Complaints by year and month, received versus sent to company, product by year |
| Channel representation | Complaints by submission channel, and channel by product |
| Geographic representation | Complaints by state |
| Baseline approach | Majority-class prediction, used as the minimum benchmark |
| Alternative approach | Keyword-based priority scoring (Low, Medium, High) |
| Product classification model | TF-IDF vectorisation with Logistic Regression (`class_weight="balanced"`), evaluation, confusion matrix and confidence-threshold analysis |
| Data critique | Limits of the dataset and how they limit our conclusions |
| Decision log | The six required decisions, with reasons |
| Business recommendation | What to do with the model, at what threshold, and how to monitor it |

## 7. Method summary

- **Target (y):** `product`
- **Feature (X):** `consumer_complaint_narrative` only. Outcome fields (`company_response_to_consumer`, `timely_response`, `consumer_disputed`) are **not** used, because they are only known after a complaint has been handled (data leakage).
- **Baseline:** always predict the most frequent product (Debt collection).
- **Model:** TF-IDF (English stop words removed) followed by Logistic Regression with balanced class weights.
- **Main metric:** macro F1, because accuracy is misleading when one product dominates. Precision, recall and accuracy are reported as supporting metrics.
- **Confidence policy:** a complaint is auto-routed only if the model's highest predicted probability is at or above a threshold. Otherwise it goes to a human reviewer.

## 8. Results

All results below come from the 500-row file (a test set of 40 complaints, 8 in Credit reporting and 32 in Debt collection).

| | Accuracy | Macro F1 |
|---|---|---|
| Majority-class baseline | 0.80 | 0.44 |
| TF-IDF + Logistic Regression | 0.775 | 0.63 |

Model macro precision is 0.64 and macro recall is 0.625. By product, the F1 score for Debt collection is 0.86 and for Credit reporting is 0.40.

**Confidence-threshold analysis (test set):**

| Threshold | Auto-routed | Sent to human | Coverage | Accuracy of auto-routed |
|---|---|---|---|---|
| 0.50 | 38 | 2 | 95.0% | 78.9% |
| 0.60 | 17 | 23 | 42.5% | 76.5% |
| 0.70 | 3 | 37 | 7.5% | 100% |
| 0.80 | 1 | 39 | 2.5% | 100% |
| 0.90 | 0 | 40 | 0% | n/a |

**Recommendation:** use the model as a **decision-support tool**, not as fully automatic routing, with staff keeping the final routing decision. Start from a 0.50 confidence threshold and monitor macro F1, precision, recall, accuracy, and the share of complaints auto-routed versus sent for human review. Retrain as new data arrives. See the notebook for the full reasoning.

## 9. Limitations (summary of the Data Critique)

- **Self-selected narratives:** only complaints whose consumers opted in to publish their text have narratives, so the model is trained on a non-random subset.
- **Class imbalance:** Debt collection makes up most of the narratives, so accuracy can hide poor performance on the smaller category.
- **Channel and geography:** the web channel dominates and states are not equally represented, so results should not be assumed to hold for other channels or regions.
- **Small sample:** the saved results use a 200-narrative sample with a 40-complaint test set, so all figures are provisional and a single complaint moves accuracy by 2.5 percentage points.
- **Scope:** the data covers US regulated financial services and a federal complaint mechanism only. Findings describe US consumers who know about and choose to use that mechanism, not consumers in general.

## 10. Future work

- Re-run the full pipeline on the complete Kaggle dataset and update all figures.
- Merge legacy product names into the consolidated categories before modelling.
- Compare the baseline against more than one alternative model on the same train/test split.
- Check performance by channel and by state.
- Validate priority scoring against a real outcome rather than keyword-generated labels.
- Put the monitoring plan into practice, including a drift check on the product mix over time.
