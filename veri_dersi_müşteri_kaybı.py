"""Leakage-safe customer churn evaluation; run with --no-plots on a server."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent


def prepare_data(df):
    df = df.copy()
    if not {'CustomerID', 'Churn'}.issubset(df.columns):
        raise ValueError('CustomerID and Churn columns are required.')
    if df['Churn'].isna().any() or set(df['Churn'].unique()) != {0, 1}:
        raise ValueError('Churn must contain both 0 and 1, with no missing labels.')
    if df['CustomerID'].isna().any() or df['CustomerID'].duplicated().any():
        raise ValueError('CustomerID must be present and unique.')
    replacements = {
        'PreferredLoginDevice': {'Phone': 'Mobile Phone'},
        'PreferredPaymentMode': {'CC': 'Credit Card', 'COD': 'Cash on Delivery'},
        'PreferedOrderCat': {'Mobile': 'Mobile Phone'},
    }
    for column, mapping in replacements.items():
        df[column] = df[column].replace(mapping)
    X = df.drop(columns=['CustomerID', 'Churn'])
    y = df['Churn'].astype(int)
    # Identical profiles (including missing-value patterns) stay in one fold.
    groups = pd.util.hash_pandas_object(X, index=False)
    return X, y, groups


def build_models():
    numeric = Pipeline([
        ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scale', StandardScaler()),
    ])
    categorical = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
        ('encode', OneHotEncoder(handle_unknown='ignore')),
    ])
    def pipeline(model):
        # Each estimator gets its own transformer; CV clones the entire pipeline.
        from sklearn.base import clone
        preprocess = ColumnTransformer([
            ('numeric', clone(numeric), make_column_selector(dtype_include=np.number)),
            ('categorical', clone(categorical), make_column_selector(dtype_exclude=np.number)),
        ])
        return Pipeline([('preprocess', preprocess), ('model', model)])
    return {
        'Dummy (majority)': pipeline(DummyClassifier(strategy='most_frequent')),
        'Logistic Regression': pipeline(LogisticRegression(max_iter=2000, random_state=42)),
        'Random Forest': pipeline(RandomForestClassifier(n_estimators=100, random_state=42)),
    }


def make_splits(X, y, groups):
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train, test = next(outer.split(X, y, groups))
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(inner.split(X.iloc[train], y.iloc[train], groups.iloc[train]))
    if set(groups.iloc[train]) & set(groups.iloc[test]):
        raise ValueError('Training and test profiles overlap.')
    for a, b in folds:
        if set(groups.iloc[train].iloc[a]) & set(groups.iloc[train].iloc[b]):
            raise ValueError('Cross-validation profiles overlap.')
    for indices in [train, test] + [train[v] for fold in folds for v in fold]:
        if y.iloc[indices].nunique() != 2:
            raise ValueError('Every evaluation partition must contain both target classes.')
    return train, test, folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluation_results.json')
    args = parser.parse_args()
    raw = pd.read_csv(ROOT / 'e_commerce_dataset_temizlenmis.csv')
    X, y, groups = prepare_data(raw)
    train, test, folds = make_splits(X, y, groups)
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
    }
    results = {'data': {
        'rows': len(X), 'features': X.shape[1],
        'positive_rows': int(y.sum()), 'missing_cells': int(X.isna().sum().sum()),
        'repeated_profiles': int(groups.duplicated().sum()),
        'train_rows': len(train), 'test_rows': len(test),
        'train_positive_rows': int(y.iloc[train].sum()),
        'test_positive_rows': int(y.iloc[test].sum()),
        'shared_train_test_profiles': 0,
        'method': 'StratifiedGroupKFold: first outer fold hold-out; five inner folds on training only',
    }, 'models': {}}
    print(json.dumps(results['data'], indent=2))
    for name, model in build_models().items():
        cv = cross_validate(model, X.iloc[train], y.iloc[train], cv=folds,
                            scoring=scoring, error_score='raise')
        model.fit(X.iloc[train], y.iloc[train])
        predicted = model.predict(X.iloc[test])
        holdout = {
            'accuracy': accuracy_score(y.iloc[test], predicted),
            'precision': precision_score(y.iloc[test], predicted, zero_division=0),
            'recall': recall_score(y.iloc[test], predicted, zero_division=0),
            'f1': f1_score(y.iloc[test], predicted, zero_division=0),
        }
        cm = confusion_matrix(y.iloc[test], predicted, labels=[0, 1])
        results['models'][name] = {
            'holdout': holdout, 'confusion_matrix': cm.tolist(),
            'cv_train': {metric: {'mean': float(cv['test_' + metric].mean()),
                                  'std': float(cv['test_' + metric].std())}
                         for metric in scoring},
        }
        print(name, json.dumps(results['models'][name], indent=2))
        if not args.no_plots:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
            plt.title(name + ' (Hold-out)')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            plt.show()
            plt.close()
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Results saved:', args.output.name)


if __name__ == '__main__':
    main()
