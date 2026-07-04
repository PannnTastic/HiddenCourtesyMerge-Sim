---
marp: true
title: HiddenCourtesyMerge-Sim
paginate: true
---

# HiddenCourtesyMerge-Sim

### Benchmark POMDP untuk Inferensi Courtesy Tersembunyi pada Merge Negotiation

Presentasi untuk Research Lead
Alur: **Pembuatan Data → Model → Temuan**

> Pertanyaan inti: *Kalau ego vehicle bisa menebak apakah pengemudi merge akan mengalah, apakah keputusannya jadi lebih aman?*

> Temuan inti: **Belief yang akurat secara offline belum tentu membuat closed-loop driving lebih aman — karena tebakan yang benar sering datang terlambat.**

---

## Peta Presentasi

1. **Motivasi & Pertanyaan Riset** — kenapa masalah ini menarik
2. **Bagian 1 — Pembuatan Data** — `highway-env merge-v0` + injeksi hidden courtesy
3. **Bagian 2 — Model** — formulasi POMDP, observation model, belief, 5 policy
4. **Bagian 3 — Temuan** — hasil, signifikansi, failure mode timing, robustness
5. **Kesimpulan, Novelty, & Limitasi**
6. **Diskusi Terbuka** — yang perlu dipertajam sebelum submit

---

## Motivasi

Saat merge di jalan tol, sebuah autonomous vehicle harus **bertindak sebelum** niat pengemudi lain benar-benar terlihat.

Asumsi umum di literatur intent-aware planning:

> "Semakin akurat estimasi niat tersembunyi, semakin aman keputusannya."

Masalahnya, asumsi ini jarang diuji secara bersih karena:

- Data naturalistik **jarang punya label niat yang terverifikasi**.
- Studi simulasi sering membandingkan policy pada **episode yang berbeda-beda**, sehingga sulit memisahkan efek inferensi dari efek keberuntungan skenario.

**Kontribusi kami: sebuah benchmark terkontrol yang menguji asumsi ini secara langsung.**

---

## Pertanyaan Riset

Kami memisahkan tiga hal yang sering dicampur:

| # | Pertanyaan | Mengukur |
|---|------------|----------|
| 1 | **Inference** — bisakah ego menebak hidden courtesy? | kualitas filter |
| 2 | **Belief-based decision** — kalau ego punya belief, apakah policy membaik? | pemanfaatan belief |
| 3 | **Closed-loop safety** — saat dijalankan, apakah collision turun? | hasil akhir |

Desain benchmark dibuat agar ketiganya bisa **diukur terpisah** lewat evaluasi *matched* (seed identik antar policy).

---

# Bagian 1
# Pembuatan Data

`highway-env merge-v0` + injeksi hidden courtesy

---

## Basis: `highway-env merge-v0`

Kami **tidak** membangun simulator dari nol. `merge-v0` sudah menyediakan:

- jalan tol dua-lajur + satu on-ramp,
- kendaraan ego dan kendaraan latar,
- dinamika kendaraan (IDM untuk longitudinal, MOBIL untuk lane-change),
- meta-action diskrit: `LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER`.

Konfigurasi episode:

- **60 policy steps** per episode, `Δt = 0.2 s` (≈ 12 detik interaksi).
- Kendaraan ramp dipilih sebagai *merge vehicle* yang jadi fokus negosiasi.

> Novelty bukan pada simulatornya — tapi pada **variabel laten** yang kami suntikkan ke dalamnya.

---

## Yang Kami Tambahkan: Hidden Courtesy

Setiap episode, *merge vehicle* diberi satu **tipe courtesy biner** yang disembunyikan dari ego:

```text
m ∈ {cooperative, non_cooperative}     P(cooperative) = P(non_cooperative) = 0.5
```

- `cooperative` → cenderung melambat, memberi ruang.
- `non_cooperative` → agresif, menjaga speed tinggi, gap kecil.

Properti desain penting:

- Tipe **fixed sepanjang episode**: `P(mₜ₊₁ = mₜ) = 1`.
- Label **direkam** untuk evaluasi ground-truth, tapi **disembunyikan** dari semua policy non-oracle.

> Desain biner + fixed sengaja dipilih untuk **mengisolasi efek timing filter** dari kerumitan intent yang berubah-ubah (non-stationary).

---

## Cara Injeksi: Parameter IDM/MOBIL

Courtesy diwujudkan **hanya** lewat rentang parameter IDM/MOBIL — tanpa override per-step, tanpa teleportasi, tanpa reward yang bergantung label.

| Parameter | Cooperative | Non-cooperative |
|---|---|---|
| Target speed (m/s) | [24.0, 28.0] | [10.0, 16.0] |
| DISTANCE_WANTED (m) | [5.0, 9.0] | [1.5, 4.0] |
| TIME_WANTED (s) | [1.4, 2.2] | [0.5, 1.0] |
| POLITENESS | [0.35, 0.65] | [0.0, 0.2] |
| COMFORT_ACC_MAX (m/s²) | [1.8, 2.8] | [3.0, 5.5] |
| COMFORT_ACC_MIN (m/s²) | [−3.0, −1.8] | [−5.5, −3.0] |

Tiap parameter di-*sample* sekali di awal episode, lalu ditanam sebagai atribut kendaraan.

---

## Dataset yang Dihasilkan

```bash
python generate_hidden_courtesy_merge_dataset.py --episodes 3000 --seed 7
```

- **3.000 episode**, di-generate dengan `belief_policy` sebagai ego.
- Split **70/15/15** (train/val/test) berdasarkan indeks episode.
- CSV menyimpan: trajektori, motion cues per step, dan label courtesy tersembunyi.

Catatan jujur untuk downstream users:

- Karena ego = `belief_policy`, distribusi collision dataset **lebih jinak** daripada distribusi naive-policy → state berisiko tinggi *underrepresented*.
- **26% episode tidak punya step di interaction window** (merge vehicle tak pernah dalam 40 m) → di episode ini belief tak pernah update dari prior.

---

## Kunci Metodologi: Matched Evaluation

Inilah yang membuat benchmark ini bersih:

```text
Semua policy dievaluasi pada n = 300 episode IDENTIK (seed 17),
berbagi: environment seed, label courtesy, densitas traffic, noise observasi.
```

Konsekuensinya:

- Perbedaan hasil **tidak** bisa dikaitkan ke "policy ini kebetulan dapat episode lebih mudah".
- Memungkinkan **paired statistical test** (McNemar untuk collision, paired-*t* untuk reward).
- Memungkinkan perbandingan **counterfactual** yang sebenarnya.

> Seed dipisah per peran: `999` kalibrasi, `7` dataset, `17` evaluasi, `42` GIF.

---

# Bagian 2
# Model

Formulasi POMDP, observation, belief, policy

---

## Kenapa POMDP?

Ada bagian dunia yang **tak terlihat langsung** oleh ego: courtesy type `m`.

- **Tanpa POMDP** → ego hanya melihat fisika: jarak, kecepatan, akselerasi.
- **Dengan POMDP** → ego menyimpan *belief*, distribusi probabilitas atas hidden state:

```text
b(cooperative) = 0.7,   b(non_cooperative) = 0.3
```

POMDP menjawab: *"Dengan ketidakpastian ini, action apa yang paling masuk akal?"*

Formulasi:

```text
M = (S, A, O, T, Z, R, γ, b₀)
```

---

## State & Action Space

**Compact POMDP state** (yang dihadapi policy):

```text
sₜ = (x_e, v_e, x_m, v_m, dₜ, Δvₜ, m)
     dₜ = x_m − x_e (jarak relatif),  Δvₜ = v_m − v_e
```

Bagian fisik teramati; courtesy `m` tidak.

**Action space** (policy terstruktur):

```text
A = {IDLE, SLOWER}
```

| Action | Arti |
|---|---|
| `IDLE` | pertahankan perilaku saat ini |
| `SLOWER` | melambat untuk membuka gap |

> Dibatasi 2 action agar perbandingan rule/belief/oracle **fair** — beda hasil murni karena **informasi courtesy**, bukan karena action space yang lebih ekspresif. (Lane-change & accelerate = limitation yang diakui.)

---

## Observation: Apa yang Dilihat Ego

Ego **tidak** melihat `m`. Ego hanya melihat 3 motion cue:

```text
oₜ = (ardₜ, mvsₜ, mvaₜ)
```

| Fitur | Arti |
|---|---|
| `ard` | absolute relative distance |
| `mvs` | merge vehicle speed |
| `mva` | merge vehicle acceleration |

Observasi hanya aktif di **interaction window**:

```text
2 m < dₜ < 40 m
```

Dari 5 kandidat fitur awal, 2 dibuang berdasarkan kriteria pra-spesifikasi:

- `urg` (urgency) — daya pisah lemah (sep. = 0.14).
- `rs` (relative speed) — hampir kolinear dengan `mvs` (Pearson r = 0.991).

---

## Observation Model: Kalibrasi

Likelihood produk-Gaussian diagonal:

```text
Z(oₜ | m) = ∏ᵢ N(o_{t,i}; μ_{m,i}, σ²_{m,i})
```

Dikalibrasi dari **60 episode/kelas** pada seed 999 (independen dari evaluasi):

| Fitur | μ coop | σ coop | μ non-coop | σ non-coop | Separasi |
|---|---|---|---|---|---|
| ard (m) | 27.50 | 9.02 | 22.76 | 10.57 | 0.48 (borderline) |
| mvs (m/s) | 9.57 | 3.88 | 13.24 | 1.68 | **1.32 (OK)** |
| mva (m/s²) | −0.94 | 1.70 | −0.05 | 0.50 | 0.81 (OK) |

> `mvs` adalah pemisah terkuat — masuk akal: cooperative melambat (≈9.6 m/s), non-cooperative jaga speed (≈13.2 m/s).

---

## Observation Model: Validasi

Pada 1.809 step validasi held-out:

| Metrik | Model | Uniform |
|---|---|---|
| Classification acc. | **98.3%** | 50.0% |
| Mean NLL (nats) | 0.168 | 0.693 |
| Brier score | 0.055 | 0.500 |
| ECE | 0.092 | — |

- Bootstrap 95% CI akurasi: **[97.5%, 98.8%]**.
- Catatan: kalibrasi tak seimbang **5.2 : 1** (290 step coop vs 1.519 non-coop) → parameter Gaussian cooperative punya ketidakpastian lebih tinggi.

> **Filter ini akurat. Tahan poin ini — justru di sinilah kejutannya nanti.**

---

## Belief Update

Karena courtesy fixed (`P(m'|m)=1` jika `m'=m`), update Bayes menyederhana:

```text
b̃ₜ₊₁(m) = η · Z(oₜ | m) · bₜ(m)          (Bayes)
bₜ₊₁    = (1 − λ)·b̃ₜ₊₁ + λ·b₀            (shrinkage,  λ = 0.08)
```

- `b₀ = [0.5, 0.5]` — netral sebelum melihat evidence.
- **Shrinkage** menarik belief sedikit ke prior tiap step → mencegah overconfidence akibat observasi antar-step yang berkorelasi.

Pemilihan `λ = 0.08` (dari sweep pada train/val):

| λ | Final-step acc. | ECE |
|---|---|---|
| 0.00 | 0.891 | 0.175 |
| **0.08** | **0.938** | **0.063** |
| 0.20 | 0.934 | 0.024 |

> `λ=0.08` = nilai terkecil yang mencapai final-step acc dalam 0.1 pp dari puncak, sambil menjaga ECE rendah.

---

## Lima Policy yang Dibandingkan

| Policy | Akses informasi | Mekanisme |
|---|---|---|
| `random` | — | action acak (baseline bawah) |
| `rule` | tidak tahu courtesy | belief dipaku `[0.5, 0.5]` |
| `belief` | belief Bayesian | filter + action table |
| `oracle` | **tahu label asli** | one-hot belief + action table sama |
| `pomcp` | belief Bayesian | **online planning** (UCT) |

Empat policy heuristik berbagi **action table V-shape** yang sama:

```text
p_SLOWER(b) = 0.735 + 0.185·|2·b_coop − 1|
```

→ paling waspada saat yakin (kedua ekstrem), paling longgar saat belief uniform.

> `oracle` = *oracle-heuristic*, **bukan** oracle optimal. Ia tahu label, tapi tetap dibatasi action table yang sama → ia ceiling untuk *table ini*, bukan ceiling POMDP.

---

## POMCP sebagai Baseline Planning

`pomcp_policy` = Partially Observable Monte Carlo Planning (Silver & Veness, 2010):

- Tiap decision step: **100 simulasi UCT**, horizon **10 step (2 s)**, `γ = 0.98`.
- Generative model menjalarkan `(dₜ, v_e, v_m)` pakai **mean IDM params** per kelas, sample `m` dari belief saat ini.
- **Penting:** reward planning POMCP berbeda dari reward evaluasi (collision zone `d<1.5 m`, tanpa survival bonus).

> POMCP = *planning over uncertainty*, bukan reaksi ke point-estimate belief. Kita perlakukan sebagai **strong approximate planner**, bukan klaim kontrol optimal.

---

## Reward Evaluasi

```text
Rₜ = r_env_t − 0.25 · 1[TTCₜ < 3s   atau  dₜ ∈ (0, 15m)]
```

dengan `r_env_t ≈ v_e / 30`.

Mengukur tradeoff:

- **throughput** (reward dari kecepatan),
- **safety** (penalti close-call: TTC kecil / jarak dekat).

> Semua policy heuristik dievaluasi di bawah reward yang sama ini → perbandingan apple-to-apple.

---

# Bagian 3
# Temuan

---

## Hasil Utama (n = 300, matched)

| Policy | Collision [%] ↓ | Reward ↑ | Success [%] | Mean min TTC [s] |
|---|---|---|---|---|
| **pomcp** | **5.0** [2.7, 7.7] | **57.36** | **95.0** | 3.14 |
| oracle | 10.3 [7.0, 14.0] | 56.46 | 89.7 | 3.03 |
| belief | 15.0 [11.0, 19.0] | 55.33 | 85.0 | 2.95 |
| rule | 18.3 [14.3, 23.0] | 54.90 | 81.7 | 3.02 |
| random | 87.0 [83.0, 91.0] | 36.71 | 13.0 | 1.88 |

Urutan: **pomcp < oracle < belief < rule ≪ random**

> Urutannya "benar" secara intuitif. Pertanyaannya: **mana yang signifikan?**

---

## Uji Signifikansi — Inilah Kejutannya

Holm–Bonferroni atas 20-test family:

| Perbandingan | Collision p_adj | Signifikan? |
|---|---|---|
| pomcp vs oracle | 0.014 (\|h\|=0.204) | ✅ Ya |
| pomcp vs rule | <0.001 (\|h\|=0.434) | ✅ Ya |
| **oracle vs rule** | **0.005** (\|h\|=0.230) | ✅ **Ya** |
| **belief vs rule** | **0.330** (\|h\|=0.090) | ❌ **Tidak** |

Dua baris terakhir adalah jantung paper:

- **oracle vs rule signifikan** → hidden courtesy memang **decision-relevant**. Kalau kamu *tahu* labelnya, collision cooperative turun drastis.
- **belief vs rule TIDAK signifikan** → padahal filter-nya 98.3% akurat. *Kenapa?*

---

## Bukan Sekadar Kurang Sampel

Hipotesis pertama: "mungkin cuma underpowered di n=300." Kami uji dengan run konfirmatori **n = 1.960** per policy:

| Policy | n | Collision | Reward |
|---|---|---|---|
| oracle | 1960 | 8.9% | 56.58 |
| belief | 1960 | **13.8%** | 55.41 |
| rule | 1960 | **15.1%** | 55.34 |

→ belief vs rule: **13.8% vs 15.1%, p_adj = 0.489** — tetap tidak signifikan, dan gap-nya malah **mengecil** (3.3 pp → 1.3 pp).

> Ukuran sampel saja tidak menjelaskan null result. Ada sesuatu yang lebih mendasar.

---

## Failure Mode: Timing Belief

Akurasi belief **per-step** di interaction window (episode non-cooperative):

| Step ke- | Akurasi belief |
|---|---|
| Step 1 | 75.3% |
| Step 2 | 71.3% |
| Steps 4–5 | 70.0% |
| Steps 11–20 | 80.7% |
| **Steps 21+** | **90.3%** |

```
Awal window (steps 1-5): ~71%   ← di sinilah keputusan gap dibuat
Akhir window (step 21+) : >90%   ← belief baru andal di sini, tapi sudah telat
```

**Mekanisme:** saat ego pertama masuk window, IDM non-cooperative **sempat melambat** merespons deteksi ego → observasinya **tumpang-tindih** dengan kelas cooperative. Belief baru reliable setelah banyak keputusan gap sudah diambil.

---

## Catatan Penting: 98.3% vs 83.6%

Dua angka akurasi yang harus dibedakan:

| Angka | Konteks | Nilai |
|---|---|---|
| Classification acc. | step-level, distribusi **kalibrasi** (seed 999) | **98.3%** |
| Final-step acc. | belief terakhir per episode (dataset) | 93.8% |
| **In-window step acc.** | **closed-loop**, belief policy (seed 7) | **83.6%** |

> Gap kalibrasi → deployment (98.3% vs 83.6%) **sendiri adalah temuan**: akurasi kalibrasi melebih-lebihkan kegunaan closed-loop. Persis tesis paper.

---

## Konfirmasi Kausal: Delayed-Oracle

Uji bersih untuk hipotesis timing: ganti belief dengan **oracle hanya di step 1–5**, sisanya pakai belief Bayesian.

```text
belief biasa     : 15.0% collision
delayed-oracle   : 11.3% collision     (p_adj = 1.0, |h| = 0.109)
```

- Arah perubahan **konsisten** dengan hipotesis timing-bottleneck.
- **Jujur:** hasil ini **tidak signifikan** → belum menutup kemungkinan action-table yang kurang sensitif. Kami laporkan apa adanya.

> Inilah probe yang tepat: kalau kita "perbaiki" justru window kritis 1–5, collision turun ke arah oracle.

---

## Robustness Checks

Apakah null result hanya artefak tuning? Tidak:

| Kekhawatiran | Cek | Hasil |
|---|---|---|
| Action table kurang sensitif? | Sweep belief gain 10× [0.05–0.50] | belief tetap **flat 15.0%**; oracle membaik → 5.7% |
| Pilihan bobot reward? | Replay w ∈ {0.10…1.00} | ranking **tak berubah** |
| Mis-spec kovarians? | Full-cov Gaussian | offline acc +5.5 pp; closed-loop **tak berubah** |
| Noise observasi? | Skala {0.25…4.00} (16×) | POMCP tetap terbaik: 4.3–5.7% |
| Bias rollout POMCP? | Ganti trigger rollout | collision tetap 5.0% → **UCT look-ahead** yang berperan |

> Belief flat di 15% meski gain dinaikkan 10× = bukti kuat: masalahnya **kapan belief tiba**, bukan **seberapa keras** belief dipakai.

---

## Kenapa POMCP Menang

POMCP tidak memperlakukan belief sebagai point-estimate yang dialirkan ke action table tetap. Ia **mencari ke depan** atas trajektori yang di-sample dari distribusi belief.

- Look-ahead-nya **robust terhadap transien likelihood early-window** yang melumpuhkan heuristik.
- Ablation rollout-trigger menunjukkan keunggulan datang dari **UCT planning**, bukan rollout yang konservatif.

> Pelajaran: ketika belief andal *terlambat*, **planning over uncertainty** mengalahkan **reaksi terhadap belief saat ini**.

---

## Kesimpulan

1. Hidden courtesy **decision-relevant** → oracle signifikan mengalahkan rule.
2. Belief Bayesian **akurat offline** (98.3%) tapi **tidak** menutup gap closed-loop.
3. Penyebabnya **timing**: belief paling lemah (≈71%) justru di window keputusan, baru andal (>90%) setelah telat.
4. **POMCP** terbaik (5.0% collision) karena merencanakan ke depan, bukan bereaksi.

> **Pelajaran metodologis:** benchmark latent-intent harus menguji bukan hanya *apakah* niat bisa di-infer, tapi *apakah inferensi tiba cukup awal* dan *dipakai policy yang cukup ekspresif* untuk mengubah hasil.

---

## Novelty yang Aman Diklaim

> HiddenCourtesyMerge-Sim adalah **benchmark terkontrol** dengan label laten terverifikasi dan evaluasi *matched-counterfactual* untuk menguji apakah inferensi hidden courtesy benar-benar membantu closed-loop merge safety.

Finding utama:

> *Accurate offline belief estimation can fail to improve closed-loop safety when the belief becomes reliable only after the safety-critical decision point.*

**Yang TIDAK kami klaim:** simulator pertama, validasi real-world, model psikologi manusia, atau POMDP terpecahkan optimal.

---

## Diskusi Terbuka — Untuk Dipertajam

Hal yang menurut saya akan disorot reviewer kuat (worth diskusi sebelum submit):

1. **Klaim "bukan sample size".** Run n=1.960 dirancang untuk efek 3.3 pp, tapi efek menyusut ke 1.3 pp → run itu *underpowered* untuk efek baru. Lebih aman: *"efek dibatasi ≤ 1.3 pp"* daripada *"sample size tak menjelaskan null"*.
2. **26% episode non-interaction** = belief ≡ rule by construction → ini *melemahkan* efek yang ingin dideteksi. Sebaiknya dihubungkan eksplisit ke null result (memperkuat tesis kita).
3. **POMCP pakai reward berbeda** (tanpa survival bonus, collision-focused) → keunggulan 5% vs 15% sebagian bisa dari *objective shaping*, bukan murni planning. Opsi: jalankan POMCP di reward evaluasi, atau caveat eksplisit di abstract.

> Ketiganya tidak mematahkan paper — tapi mempertajam batas klaim. #1 & #3 paling rawan.

---

## Reproduksi

```bash
# 1. Kalibrasi observation model
python calibrate_observation_model.py --episodes 60 --seed 999

# 2. Generate dataset kanonik
python generate_hidden_courtesy_merge_dataset.py --episodes 3000 --seed 7

# 3. Evaluasi heuristik (matched, seed 17)
python evaluate_policies.py --episodes 300 --seed 17 \
    --policies belief_policy rule_policy random_policy oracle_policy

# 4. Evaluasi POMCP (indeks episode sama)
python evaluate_policies.py --episodes 300 --seed 17 \
    --policies pomcp_policy --pomcp-sims 100 --pomcp-horizon 10

# 5. Bangun ulang tabel paper
python make_tier1_tables.py
```

Visualizer interaktif: `python sim_visualizer.py` (P/B/O/R/N untuk ganti policy).

---

# Terima Kasih

**HiddenCourtesyMerge-Sim**

Belief akurat ≠ keputusan aman, kalau belief tiba terlambat.

Pertanyaan & diskusi.
