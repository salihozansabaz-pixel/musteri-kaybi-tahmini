"""Leakage-safe customer churn evaluation; run with --no-plots on a server."""
import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
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
FEATURES = (
    'Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome',
    'PreferredPaymentMode', 'Gender', 'HourSpendOnApp',
    'NumberOfDeviceRegistered', 'PreferedOrderCat', 'SatisfactionScore',
    'MaritalStatus', 'NumberOfAddress', 'Complain',
    'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
    'DaySinceLastOrder', 'CashbackAmount',
)
CATEGORICAL = (
    'PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
    'PreferedOrderCat', 'MaritalStatus',
)
REPLACEMENTS = {
    'PreferredLoginDevice': {'Phone': 'Mobile Phone'},
    'PreferredPaymentMode': {'CC': 'Credit Card', 'COD': 'Cash on Delivery'},
    'PreferedOrderCat': {'Mobile': 'Mobile Phone'},
}


def prepare_features(df):
    """Apply the same fixed rules at training and inference; never learn statistics."""
    if df.empty or not df.columns.is_unique:
        raise ValueError('Input must have at least one row and unique column names.')
    missing = set(FEATURES) - set(df.columns)
    extra = set(df.columns) - set(FEATURES) - {'CustomerID', 'Churn'}
    if missing or extra:
        raise ValueError(f'Invalid columns. Missing: {sorted(missing)}; unexpected: {sorted(extra)}')
    X = df.loc[:, list(FEATURES)].copy()
    for column in FEATURES:
        if column not in CATEGORICAL:
            X[column] = pd.to_numeric(X[column], errors='raise')
            if np.isinf(X[column]).any():
                raise ValueError(f'{column} must contain finite numbers or missing values.')
    for column, mapping in REPLACEMENTS.items():
        X[column] = X[column].replace(mapping)
    return X


def prepare_data(df):
    df = df.copy()
    if not {'CustomerID', 'Churn'}.issubset(df.columns):
        raise ValueError('CustomerID and Churn columns are required.')
    if df['Churn'].isna().any() or set(df['Churn'].unique()) != {0, 1}:
        raise ValueError('Churn must contain both 0 and 1, with no missing labels.')
    if df['CustomerID'].isna().any() or df['CustomerID'].duplicated().any():
        raise ValueError('CustomerID must be present and unique.')
    X = prepare_features(df)
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


def save_model(model, X, path, model_name, holdout):
    """Save a fitted training-only pipeline and the schema it expects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'format_version': 1, 'pipeline': model,
        'features': list(X.columns), 'model_name': model_name,
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                     'pandas': pd.__version__, 'numpy': np.__version__, 'joblib': joblib.__version__},
        'fit_scope': 'outer training split only; hold-out excluded',
        'holdout': holdout,
    }, path)


def predict_customers(bundle, customers):
    """Predict with a trusted, previously fitted bundle without fitting any step."""
    if bundle.get('format_version') != 1 or bundle.get('features') != list(FEATURES):
        raise ValueError('Unsupported model format or feature schema.')
    if 'Churn' in customers.columns:
        raise ValueError('Prediction input must not include the Churn target column.')
    X = prepare_features(customers)
    model = bundle['pipeline']
    positive_index = list(model.classes_).index(1)
    result = pd.DataFrame({
        'churn_probability': model.predict_proba(X)[:, positive_index],
        'churn_prediction': model.predict(X).astype(int),
    })
    if 'CustomerID' in customers.columns:
        result.insert(0, 'CustomerID', customers['CustomerID'].to_numpy())
    return result


def save_figures(results, directory):
    """Render exactly the supplied hold-out metrics, without retraining."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'svg.hashsalt': 'churn-portfolio'}):
        fig, ax = plt.subplots(figsize=(6, 4.5), layout='constrained')
        cm = np.array(results['models']['Random Forest']['confusion_matrix'])
        ax.imshow(cm, cmap='Blues')
        for (row, col), value in np.ndenumerate(cm):
            ax.text(col, row, str(value), ha='center', va='center', fontsize=20,
                    color='white' if value > cm.max() / 2 else '#14263d')
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Retained', 'Churned'],
               yticklabels=['Retained', 'Churned'], xlabel='Predicted class',
               ylabel='Actual class', title='Random Forest: independent hold-out')
        fig.savefig(directory / 'confusion_matrix.svg', metadata={'Date': None})
        fig.savefig(directory / 'confusion_matrix.png', dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
        names = list(results['models'])
        positions = np.arange(len(names))
        for offset, metric, color in [(-0.18, 'f1', '#2463a8'), (0.18, 'recall', '#d9822b')]:
            values = [100 * results['models'][name]['holdout'][metric] for name in names]
            bars = ax.bar(positions + offset, values, width=0.34, label=metric.upper(), color=color)
            ax.bar_label(bars, fmt='%.1f', padding=3)
        ax.set(xticks=positions, xticklabels=names, ylim=(0, 105),
               ylabel='Score (%)', title='Churn detection: hold-out F1 and recall')
        ax.legend(frameon=False)
        fig.savefig(directory / 'model_comparison.svg', metadata={'Date': None})
        fig.savefig(directory / 'model_comparison.png', dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluation_results.json')
    parser.add_argument('--data', type=Path, default=ROOT / 'e_commerce_dataset_temizlenmis.csv')
    parser.add_argument('--model-output', type=Path, default=ROOT / 'artifacts' / 'churn_model.joblib')
    parser.add_argument('--figure-dir', type=Path, help='Save charts without opening a window.')
    args = parser.parse_args()
    raw = pd.read_csv(args.data)
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
        # The model family is fixed before looking at the hold-out results.
        if name == 'Random Forest':
            save_model(model, X, args.model_output, name, holdout)
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    if args.figure_dir:
        save_figures(results, args.figure_dir)
    print('Results saved:', args.output.name)
    print('Training-only Random Forest saved:', args.model_output)


if __name__ == '__main__':
    main()
