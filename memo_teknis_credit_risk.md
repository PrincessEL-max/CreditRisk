# Memo Teknis: Prediksi Risiko Kredit

**Proyek:** Klasifikasi risiko gagal bayar nasabah menggunakan machine learning
**Dataset:** Give Me Some Credit (Kaggle, 2011) — 150.000 nasabah, 10 fitur finansial
**Target:** `SeriousDlqin2yrs` — gagal bayar dalam 2 tahun (6,68% kelas positif)

## Masalah

Institusi keuangan perlu menilai risiko gagal bayar dari data aplikasi, dengan trade-off antara terlalu ketat (kehilangan nasabah baik) vs terlalu longgar (rugi karena gagal bayar). Data sangat imbalanced (6,68% positif), sehingga metrik dan strategi modeling standar tidak langsung berlaku.

## Metodologi

1. EDA mendalam untuk data quality (missing values, outlier, nilai placeholder)
2. Baseline: Logistic Regression & Random Forest, evaluasi accuracy vs metrik yang lebih tepat
3. Tiga pendekatan imbalance: class weighting, SMOTE, XGBoost scale_pos_weight — dibandingkan lewat PR-AUC
4. Hyperparameter tuning (StratifiedKFold + RandomizedSearchCV)
5. SHAP untuk explainability, optimasi threshold berbasis cost matrix

## Temuan Utama

| Area | Hasil |
|---|---|
| Data quality | `DebtRatio` tidak bermakna saat `MonthlyIncome` kosong (93% dari DebtRatio>100 berasal dari income kosong) |
| Accuracy vs F1 | Model dummy 93,3% akurasi vs Random Forest 93,6% — hampir sama, tapi F1 0 vs 0,276 |
| Imbalance handling | SMOTE (PR-AUC 0,267) underperform dibanding class weighting (0,364) |
| Model terbaik | XGBoost setelah tuning: PR-AUC 0,407, ROC-AUC 0,869 |
| Fitur paling berpengaruh | `RevolvingUtilizationOfUnsecuredLines`, diikuti riwayat telat bayar |
| Threshold optimal | 0,74 (bukan 0,5) — mengurangi total cost 19,3% |

## Rekomendasi

- **Gunakan XGBoost dengan hyperparameter hasil tuning** sebagai model utama
- **Threshold operasional 0,74**, bukan default 0,5 — signifikan mengurangi kesalahan yang lebih mahal (false positive berlebih di threshold default)
- **Evaluasi model dengan PR-AUC**, bukan accuracy, mengingat tingkat imbalance data
- Feature `extreme_delinquency_flag` dapat dihapus dari model produksi — terbukti redundan terhadap kolom telat-bayar yang sudah di-cap

## Keterbatasan

- Cost matrix (rasio 5:1) bersifat ilustratif; implementasi nyata perlu kalibrasi dengan data finansial aktual (nilai pinjaman, recovery rate)
- Model tidak dapat mendeteksi gagal bayar yang disebabkan kejadian mendadak di luar riwayat finansial tercatat (contoh ditemukan lewat SHAP: nasabah dengan seluruh indikator "aman" tetap gagal bayar)
- Dataset berasal dari 2011; validasi ulang diperlukan sebelum diterapkan pada data terkini

---
*Detail lengkap dan kode: lihat README.md dan notebook Minggu 1-4*
