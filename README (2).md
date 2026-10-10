# Consumer Complaint Routing System

A machine-learning project that predicts the financial product category
of a consumer complaint from its written narrative. The goal is to
support the initial routing of complaints to the appropriate team,
reducing the need to categorize every complaint manually.

**Live application:** https://consumer-complaint-routing.streamlit.app/

This is a research prototype for decision support. It does not replace
human review or make final complaint-handling decisions.

## Project Objective

Consumer complaint narratives are unstructured text. This project
explores whether Natural Language Processing (NLP) and machine learning
can use the narrative to predict the complaint's product category.

-   **Input:** A consumer complaint narrative.
-   **Target:** The `product` category.
-   **Output:** A suggested product category and the model's highest
    predicted class probability.
-   **Intended use:** Assist staff with initial complaint routing, with
    a person confirming the final decision.

## Machine Learning Workflow

### 1. Data preparation

The project uses consumer complaint data from the Consumer Financial
Protection Bureau (CFPB). The model uses two columns:

-   `consumer_complaint_narrative` --- the text used as the model input.
-   `product` --- the category the model learns to predict.

Rows without a complaint narrative cannot be used by this text-based
model. The dataset was prepared before model training. The model focuses
on complaint narratives and product categories so that information about
the final handling of a complaint is not used as an input.

### 2. Train-test split

The data was split into training and test sets using an 80/20 split and
`random_state=42`. The training set is used to fit the model; the test
set is used to evaluate its performance on data it did not train on.

### 3. Text representation and classification

The model is a scikit-learn `Pipeline` containing two steps:

1.  **TF-IDF Vectorizer:** Converts complaint text into numerical
    features and ignores common English stop words.
2.  **Logistic Regression:** Predicts the product category from those
    text features. `max_iter=1000` allows more iterations during
    fitting, and `class_weight="balanced"` helps account for differences
    in the number of examples in each class.

Keeping both steps in a single pipeline ensures that new complaint text
is transformed in the same way as the training text.

### 4. Baseline comparison

A simple baseline predicts the most frequent product category for every
test complaint. This provides a reference point for judging whether the
trained model improves on always predicting the majority class.

### 5. Evaluation

The project evaluates the model using accuracy, macro precision, macro
recall, macro F1, and a classification report. Macro F1 is particularly
useful here because it gives each class equal weight, making it easier
to see when performance on a less common category is weak.

Results from the notebook's 40-complaint test set:

  Metric                Result
  ------------------- --------
  Accuracy               0.775
  Macro precision        0.639
  Macro recall           0.625
  Macro F1               0.631
  Baseline macro F1       0.44

Per-class results:

  ------------------------------------------------------------------------
  Product           Precision         Recall       F1-score   Test support
  category                                                  
  ------------ -------------- -------------- -------------- --------------
  Credit                 0.43           0.38           0.40              8
  reporting,                                                
  credit                                                    
  repair                                                    
  services, or                                              
  other                                                     
  personal                                                  
  consumer                                                  
  reports                                                   

  Debt                   0.85           0.88           0.86             32
  collection                                                
  ------------------------------------------------------------------------

The model's macro F1 score (approximately 0.63) was higher than the
baseline macro F1 (0.44) on this test split. However, performance was
much weaker for the credit-reporting class than for debt collection. The
test set is small, so these results should be treated as an initial
evaluation rather than proof of performance on all future complaints.

### 6. Save the trained model

The complete pipeline is saved with Joblib:

``` python
import joblib

joblib.dump(model, "complaint_routing_model.pkl")
```

The saved file includes both the TF-IDF transformation and the trained
Logistic Regression classifier. The Streamlit app loads this file for
predictions, so it does not need to retrain the model whenever a user
submits a complaint.

## Streamlit Application

The app lets a user paste a complaint narrative and select **Classify
Complaint**. It then displays:

-   The model's suggested product category.
-   The highest predicted class probability.
-   A routing-suggestion message based on the app's 0.50 probability
    threshold.

If the highest probability is below 0.50, the app advises human review.
If it is at or above 0.50, the app presents the category as a routing
suggestion that still requires staff confirmation.

**Important:** A model probability is not a guarantee that a prediction
is correct and may not be a calibrated measure of confidence. The 0.50
threshold is a prototype policy, not evidence that automatic routing is
safe without oversight.

## Repository Structure

Keep the following files in the root of the GitHub repository:

``` text
financial-services-consumer-complaint-intelligence-/
├── app.py
├── requirements.txt
└── complaint_routing_model.pkl
```

-   `app.py` --- Streamlit interface and prediction logic.
-   `requirements.txt` --- Python packages installed by Streamlit
    Community Cloud.
-   `complaint_routing_model.pkl` --- trained scikit-learn pipeline
    loaded by the app.

The training notebook can also be included in the repository for
documentation and reproducibility.

Example `requirements.txt`:

``` text
streamlit
joblib
scikit-learn
numpy
```

Use the exact filename `requirements.txt` so Streamlit Community Cloud
can detect and install the dependencies.

## Run the App Locally

### 1. Clone the repository

``` bash
git clone <your-github-repository-url>
cd financial-services-consumer-complaint-intelligence-
```

Replace `<your-github-repository-url>` with the URL of your own GitHub
repository.

### 2. Install the dependencies

It is recommended to use a virtual environment.

``` bash
python -m venv .venv
```

Activate it on Windows:

``` bash
.venv\Scripts\activate
```

Install the required packages:

``` bash
python -m pip install -r requirements.txt
```

### 3. Start Streamlit

``` bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal, usually
`http://localhost:8501`.

Make sure `app.py` and `complaint_routing_model.pkl` are in the expected
repository location. If the model file is missing, the app will not be
able to load the trained pipeline.

## Deployment

The application is deployed with Streamlit Community Cloud and linked to
the GitHub repository.

General deployment steps:

1.  Push `app.py`, `requirements.txt`, and `complaint_routing_model.pkl`
    to the repository.
2.  Sign in to Streamlit Community Cloud and create an app connected to
    the repository.
3.  Select `app.py` as the app's main file.
4.  Deploy the app and check the logs if dependency installation or
    model loading fails.
5.  After making changes on GitHub, allow the app to rebuild or reboot
    it if needed.

**Live app:** https://consumer-complaint-routing.streamlit.app/

## Limitations and Responsible Use

-   **Missing narratives:** The model cannot classify a complaint from
    its narrative if no narrative is provided.
-   **Class imbalance:** The dataset contains substantially more
    examples of some product categories than others. Class weighting
    helps during training but does not replace the need for more
    representative data.
-   **Small evaluation set:** The reported test metrics are based on 40
    complaints and may vary with other samples.
-   **Uneven class performance:** The credit-reporting category had an
    F1-score of 0.40 in the reported test results, indicating that this
    class needs improvement.
-   **Limited generalization:** Results from this dataset should not be
    assumed to generalize to every complaint, organization, or
    population.
-   **Human oversight:** Staff should verify the suggested category
    before the complaint is routed. Low-probability predictions should
    be reviewed, and model performance should be monitored as more
    labeled data become available.

## Future Improvements

-   Collect and label more complaint narratives, especially for
    underrepresented categories.
-   Evaluate the model on a larger, more representative holdout dataset.
-   Review errors by product category and improve performance on
    minority classes.
-   Monitor macro F1, per-class precision and recall, and the share of
    cases referred for human review.
-   Retrain and validate the model when sufficient new labeled data are
    available.

## Technologies Used

-   Python
-   pandas
-   scikit-learn
-   TF-IDF text vectorization
-   Logistic Regression
-   Joblib
-   Streamlit
-   GitHub
-   Streamlit Community Cloud

## Disclaimer

This project is an educational machine-learning prototype for complaint
categorization and routing support. Predictions should be checked by a
human and should not be treated as final decisions or guarantees of
correctness.
