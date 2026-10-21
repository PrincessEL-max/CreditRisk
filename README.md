# Prediksi Risiko Kredit: Give Me Some Credit

Model machine learning untuk memprediksi nasabah yang akan **gagal bayar dalam 2 tahun ke depan**, lengkap dengan analisis kualitas data, penanganan imbalance, penjelasan model (SHAP), optimasi threshold berbasis biaya, dan dashboard Streamlit interaktif.

Proyek akhir mata kuliah Machine Learning. Data berasal dari kompetisi Kaggle *Give Me Some Credit* (2011): 150.000 nasabah, 10 fitur finansial, dan hanya **6,68%** yang gagal bayar.

## Hasil Utama

| Aspek | Hasil |
|---|---|
| Model terbaik | XGBoost (tuned): PR-AUC **0,407**, ROC-AUC **0,869** |
| Fitur paling berpengaruh | `RevolvingUtilizationOfUnsecuredLines`, lalu riwayat telat bayar |
| Threshold operasional | **0,74** (bukan 0,5), menurunkan total biaya **19,3%** pada asumsi cost matrix 5:1 |
| Metrik evaluasi | PR-AUC dan F1, bukan accuracy |

## Masalah

Pemberi kredit harus menyeimbangkan dua kesalahan: terlalu ketat (menolak nasabah baik) atau terlalu longgar (rugi karena gagal bayar). Karena hanya 6,68% nasabah yang gagal bayar, metrik dan strategi modeling standar tidak langsung bisa dipakai.

## Temuan

**1. `DebtRatio` bermakna hanya jika `MonthlyIncome` tercatat.**
Saat income tercatat, median `DebtRatio` 0,30 (wajar). Saat income kosong, median melonjak ke 1.159, dan 93% baris dengan `DebtRatio` > 100 berasal dari baris income kosong. Kemungkinan kolom ini dihitung dengan cara berbeda saat income tidak diketahui, jadi dua kolom ini tidak boleh ditangani terpisah.

**2. Nilai 96 dan 98 pada kolom telat bayar adalah kode, bukan hitungan.**
Ada 269 baris dengan nilai ini, dan tingkat gagal bayarnya 54,6% (rata-rata keseluruhan 6,7%). Artinya tetap sinyal kuat, tetapi harus diperlakukan sebagai kode khusus. Flag `extreme_delinquency_flag` dibuat untuk ini, namun ternyata redundan: kolom telat bayar yang di-clip ke 10 sudah otomatis menandai baris tersebut (SHAP flag = 0).

**3. Accuracy menyesatkan untuk data imbalanced.**
Model dummy yang selalu menebak "aman" mencapai akurasi 93,3%, nyaris sama dengan Random Forest (93,6%). Bedanya terlihat di recall: Random Forest hanya mendeteksi 367 dari 2.005 nasabah yang benar-benar gagal bayar (18%).

**4. SMOTE kalah dari cara yang lebih sederhana.**
PR-AUC SMOTE 0,267, lebih rendah dari class weighting (0,364) dan `scale_pos_weight` XGBoost (0,353). Setelah tuning, XGBoost naik ke 0,407.

**5. Income kosong justru berasosiasi dengan risiko lebih rendah.**
Gagal bayar 5,61% (income kosong) vs 6,95% (income tercatat). Kelompok income kosong median usianya 6 tahun lebih tua (57 vs 51), dan usia berkorelasi dengan risiko lebih rendah.

**6. Threshold optimal lebih tinggi dari 0,5, bukan lebih rendah.**
Dengan asumsi gagal bayar 5x lebih mahal dari salah tolak nasabah baik, threshold optimal 0,74. Pada threshold 0,5 model terlalu agresif menandai nasabah (false positive sangat banyak), sehingga menaikkan threshold menurunkan total biaya lebih besar daripada kenaikan false negative.

## Perbandingan Model

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression + class_weight | 0,861 | 0,364 |
| Random Forest + class_weight | 0,843 | 0,337 |
| **XGBoost (tuned)** | **0,869** | **0,407** |

Baseline (threshold 0,5):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Dummy (selalu prediksi 0) | 0,933 | 0 | 0 | 0 |
| Logistic Regression | 0,935 | 0,541 | 0,161 | 0,248 |
| Random Forest | 0,936 | 0,563 | 0,183 | 0,276 |

Model terbaik bergantung pada titik operasi: di area precision tinggi dan recall rendah, XGBoost (bahkan sebelum tuning) lebih baik dari Logistic Regression, jadi satu angka AUC tidak cukup untuk memilih model.

## Struktur Repo

| File | Fungsi |
|---|---|
| `CreditRisk.ipynb` | Notebook lengkap: EDA, preprocessing, baseline, imbalance, SHAP, threshold, demo |
| `app_credit_risk.py` | Dashboard Streamlit (2 tab) |
| `train_and_export.py` | Melatih ulang model dan menghasilkan semua file yang dibutuhkan dashboard |
| `cs-training.csv` | Dataset (150.000 baris) |
| `credit_model.json`, `model_metadata.json` | Model XGBoost terlatih, urutan fitur, median imputasi, threshold 0,74 |
| `model_comparison_credit.csv`, `pr_curve_data_credit.csv`, `shap_importance_credit.csv` | Hasil perbandingan model, data kurva PR, dan SHAP importance |
| `baseline_results_credit.csv` | Hasil baseline |
| `memo_teknis_credit_risk.md` | Ringkasan teknis singkat dan rekomendasi |

## Cara Menjalankan

Semua file yang dibutuhkan dashboard sudah ada di repo, jadi dashboard bisa langsung dijalankan.

```bash
pip install -r requirements.txt
streamlit run app_credit_risk.py
```

Untuk melatih ulang model dan membuat ulang file hasil:

```bash
python train_and_export.py
```

Untuk menjalankan notebook, pasang juga `imbalanced-learn` (dipakai untuk SMOTE dan belum ada di `requirements.txt`):

```bash
pip install imbalanced-learn jupyter
```

## Dashboard

- **Ringkasan Temuan:** tabel perbandingan model, grafik PR-AUC, kurva Precision-Recall semua model, dan SHAP importance.
- **Demo Prediksi:** masukkan data nasabah rekaan, lihat probabilitas gagal bayar, kategori (berisiko tinggi atau cenderung aman), dan 5 faktor paling berpengaruh untuk prediksi tersebut.

Catatan membaca probabilitas: model dilatih dengan `scale_pos_weight`, sehingga probabilitasnya bergeser ke atas dibanding peluang gagal bayar sebenarnya (base rate 6,68%). Gunakan sebagai skor risiko relatif dengan threshold 0,74, bukan sebagai peluang absolut.

## Keterbatasan

- Cost matrix 5:1 adalah asumsi ilustratif. Pada kasus nyata harus diganti dengan estimasi aktual (nilai pinjaman, recovery rate, dan sebagainya).
- Ada nasabah yang tampak sangat aman di atas kertas namun tetap gagal bayar, kemungkinan karena kejadian di luar data (PHK, darurat medis). Ini batas data, bukan batas model.
- Data berasal dari 2011, sehingga model perlu divalidasi ulang sebelum dipakai pada data terkini.

## Sumber Data

Kaggle, *Give Me Some Credit* (2011).
