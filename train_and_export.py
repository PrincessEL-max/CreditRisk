"""
Jalankan sekali untuk menghasilkan semua file yang dibutuhkan app.py:
- credit_model.json (model XGBoost terlatih)
- model_metadata.json (median imputasi, urutan fitur, threshold)
- model_comparison_credit.csv, pr_curve_data_credit.csv, shap_importance_credit.csv

Pastikan cs-training.csv sudah ada di folder yang sama (download dari notebook Bagian 1).
"""
import pandas as pd
import numpy as np
import warnings
import json
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import xgboost as xgb
import shap

print("Memuat data...")
df = pd.read_csv('cs-training.csv', index_col=0)

median_income_global = df.loc[df['MonthlyIncome'].notna(), 'MonthlyIncome'].median()
median_age_global = df.loc[df['age'] > 0, 'age'].median()
median_dependents_global = df['NumberOfDependents'].median()


def preprocess(df):
    df = df.copy()
    df['income_missing'] = df['MonthlyIncome'].isna().astype(int)
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(median_income_global)
    df['DebtRatio'] = df['DebtRatio'].clip(upper=10)
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(median_dependents_global)
    df.loc[df['age'] == 0, 'age'] = median_age_global
    df['RevolvingUtilizationOfUnsecuredLines'] = df['RevolvingUtilizationOfUnsecuredLines'].clip(upper=5)
    late_cols = ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate',
                 'NumberOfTime60-89DaysPastDueNotWorse']
    df['extreme_delinquency_flag'] = (df[late_cols] >= 96).any(axis=1).astype(int)
    for c in late_cols:
        df[c] = df[c].clip(upper=10)
    return df


print("Preprocessing...")
df_clean = preprocess(df)
X = df_clean.drop(columns=['SeriousDlqin2yrs'])
y = df_clean['SeriousDlqin2yrs']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

print("Melatih model final (XGBoost, parameter hasil tuning)...")
model = xgb.XGBClassifier(subsample=1.0, n_estimators=100, max_depth=4, learning_rate=0.1,
                           scale_pos_weight=scale_pos_weight, random_state=42,
                           eval_metric='logloss', n_jobs=-1)
model.fit(X_train, y_train)
proba_xgb = model.predict_proba(X_test)[:, 1]

print("Melatih model pembanding (LR, RF)...")
scaler = StandardScaler().fit(X_train)
lr_cw = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_cw.fit(scaler.transform(X_train), y_train)
proba_lr = lr_cw.predict_proba(scaler.transform(X_test))[:, 1]

rf_cw = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf_cw.fit(X_train, y_train)
proba_rf = rf_cw.predict_proba(X_test)[:, 1]

print("Menyimpan model & metadata...")
model.save_model('credit_model.json')

metadata = {
    'median_income': float(median_income_global),
    'median_age': float(median_age_global),
    'median_dependents': float(median_dependents_global),
    'feature_order': X_train.columns.tolist(),
    'best_threshold': 0.74,
}
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Menyimpan tabel perbandingan model...")
comparison = pd.DataFrame({
    'LR + class_weight': {'ROC-AUC': roc_auc_score(y_test, proba_lr),
                           'PR-AUC': average_precision_score(y_test, proba_lr)},
    'RF + class_weight': {'ROC-AUC': roc_auc_score(y_test, proba_rf),
                           'PR-AUC': average_precision_score(y_test, proba_rf)},
    'XGBoost (tuned)': {'ROC-AUC': roc_auc_score(y_test, proba_xgb),
                         'PR-AUC': average_precision_score(y_test, proba_xgb)},
}).T
comparison.to_csv('model_comparison_credit.csv')

print("Menyimpan data PR curve...")
pr_rows = []
for name, proba in [('LR + class_weight', proba_lr), ('RF + class_weight', proba_rf),
                     ('XGBoost (tuned)', proba_xgb)]:
    precision, recall, _ = precision_recall_curve(y_test, proba)
    for p, r in zip(precision, recall):
        pr_rows.append({'model': name, 'precision': p, 'recall': r})
pd.DataFrame(pr_rows).to_csv('pr_curve_data_credit.csv', index=False)

print("Menghitung & menyimpan SHAP importance...")
explainer = shap.TreeExplainer(model)
X_test_sample = X_test.sample(2000, random_state=42)
shap_values = explainer(X_test_sample)
shap_importance = pd.Series(np.abs(shap_values.values).mean(axis=0), index=X_test.columns)
shap_importance = shap_importance.sort_values(ascending=False)
shap_importance.to_csv('shap_importance_credit.csv', header=['mean_abs_shap'])

print("\nSelesai. File yang dihasilkan:")
print("  - credit_model.json, model_metadata.json")
print("  - model_comparison_credit.csv, pr_curve_data_credit.csv, shap_importance_credit.csv")
print("\nSekarang jalankan: streamlit run app.py")
