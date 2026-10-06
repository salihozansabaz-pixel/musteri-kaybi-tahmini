import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, 
                             recall_score, f1_score)

df = pd.read_csv('e_commerce_dataset_temizlenmis.csv')

df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace('Phone', 'Mobile Phone')
df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace({'CC': 'Credit Card', 'COD': 'Cash on Delivery'})
df['PreferedOrderCat'] = df['PreferedOrderCat'].replace('Mobile', 'Mobile Phone')

cols_with_missing = df.columns[df.isnull().any()].tolist()
for col in cols_with_missing:
    df[col] = df[col].fillna(df[col].median())

df = df.drop('CustomerID', axis=1)

le = LabelEncoder()
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop('Churn', axis=1)# churn sütünu drop 
y = df['Churn'] # tahmin ediceğimiz churn değerini buraya tanımlıyoruz

# HOLD-OUT 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Modellerin Tanımlanması
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

print("=== HOLD-OUT VALIDATION SONUÇLARI ===")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.2f}")
    
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name} (Hold-out)')
    plt.ylabel('Gerçek Değer')
    plt.xlabel('Tahmin Edilen(negatiflik-pozitiflik)')
    plt.show()


print("\n=== K-FOLD VALIDATION SONUÇLARI (K=5) ===")
scoring = ['accuracy', 'precision', 'recall', 'f1']

for name, model in models.items():
    cv_results = cross_validate(model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring=scoring)
    
    print(f"\nModel: {name}")
    print(f"Ortalama Accuracy: {cv_results['test_accuracy'].mean():.2f}")
    print(f"Ortalama Precision: {cv_results['test_precision'].mean():.2f}")
    print(f"Ortalama Recall: {cv_results['test_recall'].mean():.2f}")
    print(f"Ortalama F1-Score: {cv_results['test_f1'].mean():.2f}")