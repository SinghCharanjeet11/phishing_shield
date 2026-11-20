# PhishingShield Model Quality Monitoring and Retraining

## What model monitoring means

Model monitoring checks whether the deployed phishing model still performs well on new labelled URLs.

## Why retraining is needed

Attackers change their tactics. Retraining combines older training data with new labelled examples so the candidate model can learn new patterns.

## Workflow

```text
Incoming CSV
    |
    v
Validation + shared feature extraction
    |
    v
Champion evaluation on untouched holdout data
    |
    v
Old data + new data -> Random Forest candidate
    |
    v
Candidate evaluation on the same holdout data
    |
    v
Recall/F1 quality gate
    |
    v
Eligible for promotion OR champion retained
```

## Incoming data validation

The workflow loads `data/incoming/new_batch.csv`, requires `url` and `label`, removes duplicate URLs, checks missing values and labels, and extracts the 16 features with `src/features/extractor.py`.

Labels are `1` for phishing and `0` for legitimate.

## Candidate and comparison

The existing champion is loaded from `artifacts/model/model.joblib`. The candidate is trained separately and saved to `artifacts/model/candidate_model.joblib`. Both models are evaluated on the same holdout test data. URLs already present in the holdout are excluded from the new training batch to prevent data leakage.

## Quality gate

Recall is the primary metric because it measures how many phishing URLs are detected. A missed phishing URL is a false negative.

The candidate passes when:

1. Candidate recall is greater than or equal to champion recall.
2. Candidate F1-score is not lower than champion F1-score by more than `0.01`.

Passing marks the candidate **eligible for promotion**. This workflow does not replace `model.joblib` automatically.

## MLflow

The workflow tries to log the candidate run to `sqlite:///mlflow.db` under the `PhishingShield` experiment. MLflow is optional: if it is unavailable, the workflow continues.

Start the local MLflow UI with:

```text
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Generated artifacts

- `artifacts/model/model.joblib`: existing champion
- `artifacts/model/candidate_model.joblib`: retrained candidate
- `artifacts/reports/evaluation_report.json`: champion metrics
- `artifacts/reports/retraining_report.json`: candidate training details
- `artifacts/reports/quality_gate.json`: final decision

Run the complete workflow from the repository root:

```text
python -m src.monitoring.retraining
```
