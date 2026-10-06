"""Regression checks for leakage prevention and inference preprocessing."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('churn', ROOT / 'veri_dersi_müşteri_kaybı.py')
churn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(churn)


class EvaluationTests(unittest.TestCase):
    def test_profiles_never_cross_partitions(self):
        raw = pd.read_csv(ROOT / 'e_commerce_dataset_temizlenmis.csv')
        X, y, groups = churn.prepare_data(raw)
        train, test, folds = churn.make_splits(X, y, groups)
        self.assertEqual(len(train) + len(test), len(raw))
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(groups.iloc[train]) & set(groups.iloc[test]))
        self.assertGreater(groups.duplicated().sum(), 0)
        validation_rows = []
        for a, b in folds:
            self.assertFalse(set(groups.iloc[train].iloc[a]) & set(groups.iloc[train].iloc[b]))
            self.assertFalse(set(train[b]) & set(test))
            validation_rows.extend(b)
        self.assertEqual(sorted(validation_rows), list(range(len(train))))

    def test_preprocessing_learns_training_only_and_accepts_unknown_category(self):
        X = pd.DataFrame({'amount': [1., 3., np.nan, 5.], 'device': ['A', 'B', 'A', 'B']})
        y = [0, 1, 0, 1]
        for name, model in churn.build_models().items():
            with self.subTest(model=name):
                model.fit(X, y)
                imputer = model.named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute']
                np.testing.assert_allclose(imputer.statistics_, [3.])
                predictions = model.predict(pd.DataFrame({'amount': [1e9, np.nan], 'device': ['NEW', 'NEW']}))
                self.assertEqual(len(predictions), 2)
                np.testing.assert_allclose(imputer.statistics_, [3.])

    def test_missing_target_is_rejected_not_imputed(self):
        raw = pd.read_csv(ROOT / 'e_commerce_dataset_temizlenmis.csv')
        raw.loc[0, 'Churn'] = np.nan
        with self.assertRaisesRegex(ValueError, 'Churn'):
            churn.prepare_data(raw)

    def test_duplicate_customer_ids_are_rejected(self):
        raw = pd.read_csv(ROOT / 'e_commerce_dataset_temizlenmis.csv')
        raw.loc[1, 'CustomerID'] = raw.loc[0, 'CustomerID']
        with self.assertRaisesRegex(ValueError, 'CustomerID'):
            churn.prepare_data(raw)


if __name__ == '__main__':
    unittest.main()
