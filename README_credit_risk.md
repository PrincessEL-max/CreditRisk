# Prediksi Risiko Kredit — Give Me Some Credit

Proyek akhir mata kuliah Machine Learning. Tugasnya: dari data finansial nasabah, prediksi siapa yang bakal gagal bayar dalam 2 tahun ke depan. Datanya dari kompetisi Kaggle "Give Me Some Credit" (2011) — 150.000 nasabah, 10 fitur finansial (rasio utang, pendapatan, riwayat telat bayar, dst).

Awalnya kelihatan proyek yang lurus: bersihin data, latih model, selesai. Tapi begitu digali, data ini ternyata banyak "jebakan" kecil yang kalau dilewatin gitu aja bisa bikin kesimpulan salah. Aku coba dokumentasiin proses nemuin jebakan-jebakan itu, bukan cuma hasil akhirnya.

## Temuan 1: DebtRatio itu ternyata dua variabel yang nyamar jadi satu

Ini yang paling nggak kesangka. Awalnya cuma mau bikin boxplot `DebtRatio` biasa, tapi hasilnya aneh — kotaknya gepeng, kelihatannya semua nilai numpuk di 0 padahal sumbu-y-nya sampai ribuan. Coba di-clip biar plotnya kebaca, masih aneh juga.

Ternyata setelah dicek: `DebtRatio` itu wajar (median 0,29, masuk akal buat rasio utang-pendapatan) SELAMA `MonthlyIncome`-nya kecatat. Begitu `MonthlyIncome`-nya kosong, `DebtRatio` ikutan meledak — median jadi 1.159, dan 93% dari baris dengan DebtRatio di atas 100 itu justru baris yang income-nya kosong. Kemungkinan pas income nggak diketahui, kolom itu dihitung/diisi dengan cara yang beda sama sekali, bukan rasio hutang-pendapatan yang sebenarnya.

Ini penting karena artinya dua kolom yang kelihatannya independen (`DebtRatio` dan status `MonthlyIncome`) itu sebenernya saling nyambung erat, dan preprocessing-nya harus mempertimbangkan itu bareng, bukan ditangani terpisah-pisah kayak biasanya.

## Temuan lain: angka 98 bukan berarti "telat bayar 98 kali"

Tiga kolom riwayat telat bayar punya nilai 96 dan 98 yang muncul ratusan kali, jauh di atas nilai-nilai wajar lainnya (yang kebanyakan di bawah 20). Nggak masuk akal ada yang telat bayar 98 kali. Ini kemungkinan kode error/placeholder dari sistem lama, bukan angka asli.

Tapi pas dicek tingkat gagal bayar buat baris-baris ini (269 baris) — ternyata 54,6%, jauh di atas rata-rata keseluruhan (6,7%). Jadi meski angkanya kemungkinan bukan hitungan asli, baris-baris ini tetap sinyal kuat buat model. Solusinya bikin flag terpisah (`extreme_delinquency_flag`) sebelum nilai mentahnya di-cap, biar sinyalnya nggak ikut kepotong.

(Plot twist di poin 4: flag ini ternyata redundan — dijelasin di bawah)

## 2: accuracy itu bohong buat data kayak gini

Model dummy yang cuma selalu nebak "aman" dapet akurasi 93,3%. Random Forest yang beneran dilatih cuma naik ke 93,6% — beda tipis banget. Tapi kalau dilihat dari confusion matrix, bedanya jauh: dari 2.005 nasabah yang beneran bakal gagal bayar, cuma 367 (18%) yang kedeteksi RF. Lebih dari 8 dari 10 nasabah berisiko lolos.

Ini bukan model yang jelek, ini threshold default (0,5) yang nggak cocok buat data yang cuma 6,68% positif.

## 3: SMOTE ternyata kalah dari cara yang lebih sederhana

Nyoba 3 pendekatan buat handle imbalance: class weighting, SMOTE, sama scale_pos_weight bawaan XGBoost. Yang lumayan bikin surprised: SMOTE (PR-AUC 0,267) malah paling jelek, kalah dari class weighting biasa (PR-AUC 0,364) dan XGBoost (0,353). Awalnya kirain SMOTE bakal paling bagus karena "lebih canggih", ternyata enggak.

Setelah XGBoost di-tuning (StratifiedKFold, RandomizedSearchCV), baru naik jadi PR-AUC 0,407 — jadi yang terbaik. Tapi dari PR curve-nya kelihatan juga: di titik precision tinggi/recall rendah, XGBoost bahkan versi belum di-tuning udah lebih bagus dari LR — jadi "model terbaik" itu sebenernya tergantung mau dioperasikan di titik mana, nggak bisa dijawab cuma dari satu angka AUC.

## 4: dua hal yang paling menarik dari SHAP

**Pertama**, fitur `extreme_delinquency_flag` yang dibikin capek-capek di poin 2 ternyata SHAP importance-nya kurang lebih nol. Setelah ditelusuri, ternyata karena kolom telat-bayar yang di-clip ke maksimum 10 udah otomatis "menandai" baris-baris itu (karena nilai aslinya yang 96/98 pasti kena clip ke 10), jadi model nggak butuh flag terpisah buat tau baris itu spesial. Bukan kesalahan fatal, tapi ngasih pelajaran: dua keputusan preprocessing yang kelihatannya independen bisa saling tumpang tindih efeknya.

**Kedua**, ada pola yang awalnya kelihatan aneh: `income_missing=1` itu SHAP-nya konsisten NEGATIF (nurunin risiko), padahal intuisi awal "kalau nggak tau incomenya harusnya kelihatan lebih berisiko". Dicek ke data aslinya — ternyata memang nyata, bukan model yang salah: tingkat gagal bayar kelompok income-kosong itu 5,61%, lebih rendah dari kelompok income-tercatat (6,95%). Kemungkinan penjelasannya: kelompok income-kosong itu median usianya 6 tahun lebih tua (57 vs 51), dan usia udah kebukti berkorelasi sama risiko lebih rendah.

Yang juga menarik: kelompok ini `DebtRatio`-nya rusak parah (median 1.159, dari temuan poin 1), tapi model tetap ngasih prediksi yang masuk akal buat mereka — kelihatannya model cukup ngandelin kombinasi fitur lain, nggak ketipu sepenuhnya sama satu fitur yang datanya berantakan.

## Optimasi threshold — hasilnya kebalikan dari dugaan awal

Pakai cost matrix ilustratif (gagal bayar 5x lebih mahal dari salah tolak nasabah baik), threshold optimal yang ketemu itu **0,74** — lebih tinggi dari 0,5, bukan lebih rendah. Awalnya mikir "kan FN lebih mahal, harusnya threshold diturunin biar nangkep lebih banyak yang berisiko". Ternyata di threshold 0,5, model ini udah kelewat agresif nge-flag orang (FP sampai 5.682!), jadi meski FN naik pas threshold dinaikin, penurunan FP yang jauh lebih besar tetap menang secara total cost. Pengurangan cost totalnya 19,3%.

## Batasan yang perlu diakui

- Cost matrix (5:1) itu asumsi ilustratif, bukan angka riil dari data pinjaman sungguhan — di proyek beneran harus diganti sama estimasi aktual (rata-rata nilai pinjaman, recovery rate, dst)
- Ada kasus nasabah yang di atas kertas kelihatan aman banget (utilization rendah, nol riwayat telat) tapi tetap gagal bayar — kemungkinan karena kejadian yang nggak tertangkap di data ini (PHK, darurat medis, dst). Ini batas dari data yang tersedia, bukan sesuatu yang bisa diperbaiki cuma dengan model yang lebih canggih.
- Dataset ini dari 2011 — pola ekonomi/finansial bisa berubah, model ini nggak otomatis valid buat data masa kini tanpa divalidasi ulang

## Cara jalanin

1. `pip install pandas numpy scikit-learn imbalanced-learn xgboost shap matplotlib`
2. Notebook bisa download datanya sendiri dari mirror GitHub (link ada di dalam notebook CreditRisk.ipynb, nggak perlu akun Kaggle
