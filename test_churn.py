"""Regression checks for leakage prevention and inference preprocessing."""
import importlib.util
from pathlib import Path
import unittest
import tempfile

import joblib
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


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import churn as implementation
        cls.implementation = implementation
        # Small training fixture; persistence does not need a full CV run.
        cls.raw = pd.read_csv(ROOT / 'e_commerce_dataset_temizlenmis.csv').iloc[:80].copy()
        cls.X, cls.y, _ = implementation.prepare_data(cls.raw)
        cls.model = implementation.build_models()['Random Forest'].fit(cls.X, cls.y)

    def test_round_trip_preserves_predictions_and_probabilities(self):
        implementation = self.implementation
        customers = self.raw.drop(columns='Churn').iloc[:3]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.joblib'
            implementation.save_model(self.model, self.X, path, 'Random Forest', {})
            bundle = joblib.load(path)
            actual = implementation.predict_customers(bundle, customers)
        expected_X = implementation.prepare_features(customers)
        np.testing.assert_array_equal(actual['churn_prediction'], self.model.predict(expected_X))
        np.testing.assert_allclose(actual['churn_probability'], self.model.predict_proba(expected_X)[:, 1])
        np.testing.assert_array_equal(actual['CustomerID'], customers['CustomerID'])

    def test_inference_applies_training_category_aliases(self):
        implementation = self.implementation
        aliases = self.raw.drop(columns='Churn').iloc[:2].copy()
        aliases['PreferredLoginDevice'] = 'Phone'
        aliases['PreferredPaymentMode'] = 'CC'
        aliases['PreferedOrderCat'] = 'Mobile'
        canonical = aliases.copy()
        canonical['PreferredLoginDevice'] = 'Mobile Phone'
        canonical['PreferredPaymentMode'] = 'Credit Card'
        canonical['PreferedOrderCat'] = 'Mobile Phone'
        pd.testing.assert_frame_equal(implementation.prepare_features(aliases), implementation.prepare_features(canonical))

    def test_missing_features_and_infinity_are_rejected(self):
        implementation = self.implementation
        with self.assertRaisesRegex(ValueError, 'Missing'):
            implementation.prepare_features(self.raw.drop(columns='Tenure'))
        invalid = self.raw.copy()
        invalid.loc[invalid.index[0], 'Tenure'] = np.inf
        with self.assertRaisesRegex(ValueError, 'finite'):
            implementation.prepare_features(invalid)

    def test_target_column_is_rejected_during_inference(self):
        implementation = self.implementation
        bundle = {'format_version': 1, 'features': list(implementation.FEATURES), 'pipeline': self.model}
        with self.assertRaisesRegex(ValueError, 'target'):
            implementation.predict_customers(bundle, self.raw)

    def test_unknown_category_and_missing_value_survive_saved_model(self):
        implementation = self.implementation
        customers = self.raw.drop(columns=['Churn', 'CustomerID']).iloc[:1].copy()
        customers['PreferredLoginDevice'] = 'Previously unseen device'
        customers['Tenure'] = np.nan
        bundle = {'format_version': 1, 'features': list(implementation.FEATURES), 'pipeline': self.model}
        result = implementation.predict_customers(bundle, customers)
        self.assertEqual(result.shape, (1, 2))
        self.assertTrue(np.isfinite(result['churn_probability']).all())
        self.assertTrue(result['churn_probability'].between(0, 1).all())


if __name__ == '__main__':
    unittest.main()
