# Dataset provenance and sharing status

The bundled file is `e_commerce_dataset_temizlenmis.csv` (5,630 rows,
20 columns including `CustomerID` and `Churn`). Its download receipt, original
workbook, and transformation script were not present in the existing repository.

## Candidate upstream source

The schema and row count are consistent with **Ecommerce Customer Churn
Analysis and Prediction**, published by **Ankit Verma** on Kaggle:

https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction

This is a candidate attribution, not a verified chain of provenance. The Kaggle
page could be located on 3 November 2026, but its dataset-specific license and
original file could not be independently inspected in this review. A notebook's
Apache license or another repository's license does not establish the dataset's
license. Do not relabel this CSV as CC0 based only on third-party descriptions.

## Before publishing the data or trained artifacts

1. Recover the original download link or workbook and compare the sheet's
   columns, row count, values, missingness, and target counts against this CSV.
2. Record the dataset version, retrieval date, publisher, exact license URL,
   and all local transformations. Add a file hash for the original download.
3. Confirm that the license permits the intended redistribution. If it does
   not, keep the data out of a public repository and document how users can
   obtain and prepare their own local copy with `train.py --data PATH`.
4. Review Git history before changing repository visibility: removing a CSV
   in a new commit does not remove it from previous commits.

The repository remains private while these points are unresolved. No dataset
license or code license has been invented as part of the portfolio update.

## Interpretation limits

The feature measurement times and the exact operational definition of churn
are unverified. The experiment is an educational comparison on a fixed table;
it does not demonstrate future-customer performance or a causal retention
strategy. `examples/customers.csv` contains two manually invented examples,
not copied customer records or validation observations.
