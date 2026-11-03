# Validation record

Date: 3 November 2026. Platform: Windows, Python 3.12. Exact installed packages
are recorded in `requirements-lock.txt`. These are local checks; the workflow
badge links separately to GitHub's hosted run history.

- `python -W error -m unittest -v test_churn.py`: **9 tests passed**.
- `python -W error train.py --no-plots --output evaluation_results.json --figure-dir docs/figures`:
  all three models evaluated, JSON and plots written, Random Forest pipeline saved.
- `python -W error predict.py --input examples/customers.csv`:
  saved model loaded in a separate process and predictions written for both
  invented customers (probabilities 0.08 and 0.07; both predicted class 0).
- All recorded hold-out metrics and confusion matrices reproduce the previous
  validated report. The Random Forest remains at 94.40% accuracy, 81.52% F1,
  and 73.16% recall.

Tests cover split separation, training-only preprocessing, invalid labels,
duplicate IDs, persisted predictions/probabilities, shared category aliases,
missing-feature and infinite-value rejection, target rejection at inference,
and prediction with unseen categories and missing values.

Generated model binaries and prediction CSV files are ignored by Git. Model
files can be regenerated with `train.py`; no binary artifact is committed.
