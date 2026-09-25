import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="Prediksi Risiko Kredit", layout="wide", page_icon="💳")

st.title("💳 Prediksi Risiko Kredit")
st.caption("Proyek akhir Machine Learning — dataset Give Me Some Credit (Kaggle)")


@st.cache_resource
def load_model_and_meta():
    model = xgb.XGBClassifier()
    model.load_model("credit_model.json")
    with open("model_metadata.json") as f:
        meta = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, meta, explainer


@st.cache_data
def load_summary_files():
    comparison = pd.read_csv("model_comparison_credit.csv", index_col=0)
    pr_curve = pd.read_csv("pr_curve_data_credit.csv")
    shap_importance = pd.read_csv("shap_importance_credit.csv", index_col=0)
    return comparison, pr_curve, shap_importance


try:
    model, meta, explainer = load_model_and_meta()
    comparison, pr_curve, shap_importance = load_summary_files()
except FileNotFoundError as e:
    st.error(
        f"File belum lengkap: {e}. Jalankan `python train_and_export.py` dulu "
        "di folder yang sama dengan `cs-training.csv`, baru jalankan dashboard ini."
    )
    st.stop()

tab1, tab2 = st.tabs(["📊 Ringkasan Temuan", "🔮 Demo Prediksi"])

# ---------------------------------------------------------------------------
with tab1:
    st.header("Perbandingan Model")

    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)
        best_model = comparison["PR-AUC"].idxmax()
        st.success(f"Model terbaik: **{best_model}** (PR-AUC {comparison.loc[best_model, 'PR-AUC']:.3f})")

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        comparison["PR-AUC"].plot(kind="bar", ax=ax, color="#2980b9")
        ax.set_ylabel("PR-AUC")
        ax.set_title("PR-AUC per Model")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

    st.header("Precision-Recall Curve")
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, group in pr_curve.groupby("model"):
        ax.plot(group["recall"], group["precision"], label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.set_title("Precision-Recall Curve — Semua Model")
    plt.tight_layout()
    st.pyplot(fig)

    st.header("Fitur Paling Berpengaruh (SHAP)")
    fig, ax = plt.subplots(figsize=(9, 5))
    shap_importance["mean_abs_shap"].sort_values().plot(kind="barh", ax=ax, color="#27ae60")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Rata-rata Dampak Tiap Fitur terhadap Prediksi")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption(
        "RevolvingUtilizationOfUnsecuredLines jauh paling dominan — rasio pemakaian kredit yang tinggi "
        "adalah sinyal risiko paling kuat di model ini."
    )

# ---------------------------------------------------------------------------
with tab2:
    st.header("Coba Prediksi Sendiri")
    st.caption("Masukkan data finansial nasabah (rekaan), lihat prediksi risiko gagal bayar beserta alasannya.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Usia", min_value=18, max_value=100, value=35)
        income_unknown = st.checkbox("Pendapatan tidak diketahui")
        monthly_income = st.number_input(
            "Pendapatan bulanan (satuan asli dataset)",
            min_value=0, value=5000, disabled=income_unknown,
        )
        revolving_util = st.slider("Rasio pemakaian kredit (RevolvingUtilization)", 0.0, 2.0, 0.3, 0.01)

    with col2:
        debt_ratio = st.number_input("Debt Ratio", min_value=0.0, value=0.3, step=0.05)
        n_credit_lines = st.number_input("Jumlah kredit/pinjaman aktif", min_value=0, value=6)
        n_real_estate = st.number_input("Jumlah pinjaman properti", min_value=0, value=1)
        n_dependents = st.number_input("Jumlah tanggungan", min_value=0, value=0)

    with col3:
        late_30_59 = st.number_input("Telat bayar 30-59 hari (berapa kali)", min_value=0, value=0)
        late_60_89 = st.number_input("Telat bayar 60-89 hari (berapa kali)", min_value=0, value=0)
        late_90 = st.number_input("Telat bayar 90+ hari (berapa kali)", min_value=0, value=0)

    if st.button("Prediksi", type="primary"):
        income_missing_flag = int(income_unknown)
        income_value = meta["median_income"] if income_unknown else monthly_income
        extreme_flag = int(max(late_30_59, late_60_89, late_90) >= 96)

        data_nasabah = {
            "RevolvingUtilizationOfUnsecuredLines": min(revolving_util, 5),
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": min(late_30_59, 10),
            "DebtRatio": min(debt_ratio, 10),
            "MonthlyIncome": income_value,
            "NumberOfOpenCreditLinesAndLoans": n_credit_lines,
            "NumberOfTimes90DaysLate": min(late_90, 10),
            "NumberRealEstateLoansOrLines": n_real_estate,
            "NumberOfTime60-89DaysPastDueNotWorse": min(late_60_89, 10),
            "NumberOfDependents": n_dependents,
            "income_missing": income_missing_flag,
            "extreme_delinquency_flag": extreme_flag,
        }
        row = pd.DataFrame([data_nasabah])[meta["feature_order"]]
        proba = model.predict_proba(row)[0, 1]
        threshold = meta["best_threshold"]
        is_risky = proba >= threshold

        st.markdown("---")
        col1, col2 = st.columns(2)
        col1.metric("Probabilitas gagal bayar", f"{proba:.1%}")
        col2.metric("Kategori", "⚠️ BERISIKO TINGGI" if is_risky else "✅ Cenderung aman")

        sv = explainer(row)
        top_factors = pd.Series(sv.values[0], index=row.columns).sort_values(key=abs, ascending=False).head(5)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        colors = ["#c0392b" if v > 0 else "#27ae60" for v in top_factors.values]
        top_factors.sort_values().plot(kind="barh", ax=ax, color=colors[::-1])
        ax.set_xlabel("SHAP value (kanan = menaikkan risiko, kiri = menurunkan)")
        ax.set_title("5 Faktor Paling Berpengaruh untuk Prediksi Ini")
        plt.tight_layout()
        st.pyplot(fig)

        st.caption(
            f"Threshold operasional: {threshold} (hasil optimasi cost, lihat notebook Bagian 4). "
            "Data di atas rekaan untuk demonstrasi, bukan data nasabah sungguhan."
        )

st.markdown("---")
st.caption("Dataset: Give Me Some Credit (Kaggle, 2011) · Proyek akhir Machine Learning")
