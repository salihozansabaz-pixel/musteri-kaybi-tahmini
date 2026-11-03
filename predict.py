"""Batch prediction from a trusted local model; no training occurs here."""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from churn import ROOT, predict_customers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'artifacts' / 'churn_model.joblib',
                        help='Only load models you created or trust: joblib can execute code.')
    parser.add_argument('--input', type=Path, required=True, help='CSV with all 18 features; no Churn column.')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts' / 'predictions.csv')
    args = parser.parse_args()
    try:
        result = predict_customers(joblib.load(args.model), pd.read_csv(args.input))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f'Prediction failed: {error}\n')
    print(result.to_string(index=False))
    print('Predictions saved:', args.output)


if __name__ == '__main__':
    main()
