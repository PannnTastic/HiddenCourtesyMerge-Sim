# Memahami HiddenCourtesyMerge-Sim — Panduan Visual

> Dokumen ini menjelaskan proyek dari **dataset → model POMDP → state → observation → transition → reward → policy → temuan**, dengan fokus khusus pada **apa itu oracle dan kenapa bisa kalah**, serta **kenapa proyek ini novelty**. Visualisasi pakai diagram ASCII supaya bisa dibaca langsung. Semua angka diverifikasi dari kode & paper.

---

## Daftar Isi

1. [Cerita Besar (analogi)](#1-cerita-besar)
2. [Dataset — dari mana datanya](#2-dataset)
3. [Kenapa POMDP](#3-kenapa-pomdp)
4. [State (S)](#4-state)
5. [Observation (O) & Observation Model (Z)](#5-observation)
6. [Belief Update](#6-belief-update)
7. [Transition (T)](#7-transition)
8. [Reward (R)](#8-reward)
9. [Lima Policy](#9-policy)
10. [⭐ ORACLE: apa itu & kenapa kalah](#10-oracle)
11. [Temuan Utama](#11-temuan)
12. [Kenapa Novelty](#12-novelty)
13. [Peta Aset Visual (GIF & figur)](#13-aset-visual)

---

<a name="1-cerita-besar"></a>
## 1. Cerita Besar

Bayangkan kamu nyetir di tol. Ada mobil dari ramp mau menyelip. Kamu butuh tahu: **dia bakal ngalah atau maksa masuk?** Kamu tidak bisa baca pikiran sopirnya — cuma bisa **menebak dari gerakannya**.

```
                          MOBIL EGO (kita)
                               │
        Lajur utama   ━━━━━━━━━▼━━━━━━━━━━━━━━━━━━━━━━━━━━▶
                                    ╲
                                     ╲  ← mobil merge (sifat TERSEMBUNYI)
                       Ramp ━━━━━━━━━━╲━━━━━━━
                                       ▲
                              cooperative? atau non-cooperative?
                              (ngalah)         (ngotot)
```

**Pertanyaan riset:**
> Kalau ego bisa *menebak akurat* sifat sopir lain, apakah keputusannya jadi lebih aman?

**Jawaban (yang bikin menarik):**
> Belum tentu. Tebakan akurat bisa GAGAL menambah keselamatan — karena tebakan benar sering datang **terlambat**, setelah keputusan penting sudah diambil.

---

<a name="2-dataset"></a>
## 2. Dataset — dari mana datanya

### Simulator bukan bikinan sendiri

Pakai `highway-env merge-v0` yang **sudah** menyediakan jalan, ramp, mobil, fisika, dan aksi. **Novelty BUKAN bikin simulator.** Yang ditambahkan: **sifat tersembunyi** pada mobil merge.

### Cara nyuntik sifat: lewat parameter, bukan sulap

```
  Awal episode
       │
       ▼
  ┌──────────────────────────────────────────────┐
  │  UNDI sifat:  cooperative  /  non_cooperative │  (50% : 50%)
  └──────────────────────────────────────────────┘
       │
       ▼
  ┌──────────────────────────────────────────────┐
  │  UNDI 6 parameter IDM/MOBIL dari rentang tipe │
  │  → tanam ke mobil sebagai atribut             │
  └──────────────────────────────────────────────┘
       │
       ▼
  Simulator jalan SENDIRI (tanpa campur tangan per-langkah)
  → perilaku "ngalah/ngotot" muncul EMERGENT dari fisika
```

Contoh rentang parameter:

| Parameter        | Cooperative | Non-cooperative |
|------------------|-------------|-----------------|
| Target kecepatan | 24–28 m/s   | 10–16 m/s       |
| Jarak diinginkan | 5–9 m       | 1.5–4 m         |
| Politeness       | 0.35–0.65   | 0.0–0.2         |

### Angka dataset

```
  3.000 episode  ×  60 langkah (Δt=0.2s, ≈12 detik)
  ├── 50% cooperative, 50% non-cooperative
  ├── label disimpan untuk evaluasi, DISEMBUNYIKAN dari policy
  ├── split 70/15/15 (train/val/test)
  └── ⚠ 26% episode TANPA momen interaksi (mobil tak pernah <40m)
       → di episode ini belief TAK PERNAH update
```

---

<a name="3-kenapa-pomdp"></a>
## 3. Kenapa POMDP

MDP biasa = kamu tahu semua. Di sini ada yang **tak terlihat**: sifat `m`. Maka pakai **POMDP** (Partially Observable MDP).

```
  TANPA POMDP                      DENGAN POMDP
  ───────────                      ────────────
  ego cuma lihat angka fisik:      ego SIMPAN belief:
    jarak, kecepatan, akselerasi     P(cooperative)     = 0.7
                                      P(non_cooperative) = 0.3
                                    "aku 70% yakin dia ngalah"
```

Formal: `M = (S, A, O, T, Z, R, γ, b₀)` — kita bahas tiap huruf di bawah.

---

<a name="4-state"></a>
## 4. State (S)

Dua level state:

```
  ┌─────────────────────────────────────────────────────────┐
  │ STATE PENUH SIMULATOR (highway-env)                       │
  │  posisi SEMUA mobil, lane, geometri, param IDM, traffic…  │
  │  → dipakai simulator menjalankan dunia                    │
  └─────────────────────────────────────────────────────────┘
                          │ disederhanakan jadi
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │ STATE RINGKAS POMDP (yang dilihat policy)                 │
  │  sₜ = (x_e, v_e, x_m, v_m, dₜ, Δvₜ, m)                    │
  └─────────────────────────────────────────────────────────┘
```

| Simbol         | Arti                       | Terlihat? |
|----------------|----------------------------|-----------|
| x_e, v_e       | posisi & kecepatan ego     | ✅        |
| x_m, v_m       | posisi & kecepatan merge   | ✅        |
| dₜ = x_m − x_e | jarak relatif              | ✅        |
| Δvₜ = v_m − v_e| kecepatan relatif          | ✅        |
| **m**          | **sifat courtesy**         | ❌ TERSEMBUNYI |

---

<a name="5-observation"></a>
## 5. Observation (O) & Observation Model (Z)

Ego tak lihat `m`. Ego cuma lihat **3 petunjuk gerakan**:

```
  oₜ = (ard, mvs, mva)
        │     │     └── mva = akselerasi mobil merge
        │     └──────── mvs = KECEPATAN mobil merge  ★ petunjuk terkuat
        └────────────── ard = jarak relatif absolut
```

> Dari 5 kandidat awal, 2 dibuang: `urg` (daya beda lemah) & `rs` (kembar dgn mvs, r=0.991).

Observasi **cuma aktif di interaction window**:

```
   d=2m                                            d=40m
    │◄───────── INTERACTION WINDOW ─────────────►│
    │   (belief di-update di sini saja)           │
  terlalu                                       terlalu
  dekat                                          jauh
```

### Observation Model (Z): petunjuk → probabilitas

Z menjawab: *"Kalau cooperative, seberapa mungkin lihat gerakan ini? Kalau non-cooperative?"*

Pakai **produk Gaussian** (kurva lonceng per kelas, per fitur):

```
  Contoh fitur mvs (kecepatan merge):

  prob │      cooperative          non-cooperative
       │        ╱╲                      ╱╲
       │       ╱  ╲                    ╱  ╲
       │      ╱    ╲                  ╱    ╲
       │     ╱      ╲                ╱      ╲
       │    ╱        ╲______________╱        ╲
       └────┴─────────┴──────┬──────┴─────────┴──── kecepatan
          ~9.6 m/s         13 m/s?           ~13.2 m/s
          (melambat)        ↑                (jaga speed)
                      lihat 13 m/s → "lebih cocok NON-cooperative"
```

**Model ini AKURAT: 98.3% akurasi klasifikasi.** ⬅️ INGAT ANGKA INI — di sinilah kejutannya.

```
  Validasi observation model (n=1.809 step):
  ┌───────────────────┬────────┬─────────┐
  │ Metrik            │ Model  │ Tebak acak │
  ├───────────────────┼────────┼─────────┤
  │ Akurasi klasifikasi│ 98.3% │  50.0%  │
  │ Brier score       │ 0.055  │  0.500  │
  └───────────────────┴────────┴─────────┘
```

---

<a name="6-belief-update"></a>
## 6. Belief Update

Karena sifat tetap sepanjang episode, update Bayes sederhana:

```
  b̃ₜ₊₁(m) = η · Z(oₜ|m) · bₜ(m)       ← kalikan tebakan lama × likelihood
  bₜ₊₁    = (1−λ)·b̃ₜ₊₁ + λ·b₀         ← tarik sedikit ke netral (λ=0.08)

  mulai dari b₀ = [0.5, 0.5]  (netral)
```

```
  Ilustrasi belief menguat seiring waktu:

  P(benar) │                              ___________ >90%
    1.0    │                        _____/
           │                  _____/
    0.71   │ ___________ _____/   ← awal cuma ~71%!
    0.5 ───┼─/─────────────────────────────────────
           └──┬────┬────┬────┬────┬────┬──── langkah window
              1    5    10   15   20   25
              ▲                    ▲
        KEPUTUSAN dibuat     belief baru andal
        di sini (telat!)     (sudah terlambat)
```

---

<a name="7-transition"></a>
## 7. Transition (T)

```
  T(s'|s,a) = δ(m'=m) · T_env(s'_obs | s_obs, a, m)
              └─────┬────┘ └──────────┬──────────┘
         sifat TETAP            fisik dijalankan simulator

  ┌────────────────┐         ┌────────────────────────┐
  │ Bagian tersembunyi │     │ Bagian fisik             │
  │ cooperative →      │     │ ego SLOWER → ego melambat│
  │   tetap cooperative│     │   → gap membesar         │
  │ (P=1, tak berubah) │     │ coop → kasih gap aman    │
  └────────────────┘         │ non-coop → gap kecil     │
                             └────────────────────────┘
```

---

<a name="8-reward"></a>
## 8. Reward (R)

```
  Rₜ = r_env − 0.25 · 1[TTC<3s ATAU jarak∈(0,15m)]
       └──┬──┘   └──────────────┬───────────────┘
       throughput            penalti safety
      (≈ v_e/30)         (kalau terlalu dekat/bahaya)
```

Mengukur **tradeoff**: ego harus cepat TAPI aman. Semua policy heuristik dinilai reward yang **sama** → adil.

---

<a name="9-policy"></a>
## 9. Lima Policy

```
  random ──► rule ──► belief ──► oracle ──► pomcp
  (acak)    (buta)   (Bayesian)  (tahu     (planning
                                  label)    ke depan)
   makin ke kanan = makin banyak info / makin pintar
```

| Policy   | Tahu apa?            | Caranya                         |
|----------|----------------------|---------------------------------|
| `random` | —                    | aksi acak (baseline bawah)      |
| `rule`   | tidak tahu courtesy  | belief dipaku `[0.5,0.5]`       |
| `belief` | belief Bayesian      | filter + tabel aksi             |
| `oracle` | **tahu label asli**  | belief one-hot + tabel aksi SAMA|
| `pomcp`  | belief Bayesian      | **planning multi-langkah**      |

4 policy heuristik berbagi **satu tabel aksi V-shape**:

```
  p_SLOWER(b) = 0.735 + 0.185·|2·b_coop − 1|

  peluang │ 0.92 ●                           ● 0.92
  ngerem  │      ╲                           ╱
          │       ╲                         ╱
          │        ╲                       ╱
    0.735 │         ╲_________●__________╱   ← rule di sini
          └──────────┴────────┴────────┴──────── keyakinan
        yakin       yakin     uniform   yakin
       non-coop     ───       (0.5)     coop
                    bentuk V (simetris!)
```

> Kenapa cuma {IDLE, SLOWER}? Supaya **adil** — beda hasil murni dari INFO courtesy, bukan dari aksi ekstra (pindah lane/ngebut).

---

<a name="10-oracle"></a>
## 10. ⭐ ORACLE: apa itu & kenapa KALAH

**Ini jantung paper & pertanyaan supervisor.**

### Apa itu oracle?

Oracle = policy yang **dikasih jawaban sempurna**. Belief-nya selalu one-hot:
- tahu cooperative → `[1.0, 0.0]`
- tahu non-cooperative → `[0.0, 1.0]`

TAPI ia tetap **dipaksa** lewat **tabel aksi V-shape yang sama**.

```
  ┌─────────────────────────────────────────────────────┐
  │  Oracle = "ORACLE OF STATE" (tahu kondisi)            │
  │           BUKAN "oracle of action" (tahu aksi terbaik)│
  └─────────────────────────────────────────────────────┘
```

### Kenapa itu bikin oracle lemah? Masukkan angka:

```
  ┌──────────────────────┬────────┬──────────┬───────────┐
  │ Policy               │ b_coop │ |2b−1|   │ p_SLOWER  │
  ├──────────────────────┼────────┼──────────┼───────────┤
  │ rule                 │  0.5   │   0      │  0.735    │
  │ oracle TAHU coop     │  1.0   │   1      │  0.920 ◄┐ │
  │ oracle TAHU non-coop │  0.0   │   1      │  0.920 ◄┘ │
  └──────────────────────┴────────┴──────────┴───────────┘
                                                    │
              IDENTIK! Oracle ngerem 92% di KEDUA tipe
```

```
  💡 Masalah: tabel V SIMETRIS.
     Info "aku tahu coop" vs "aku tahu non-coop"
     → HASILNYA AKSI YANG SAMA.

     Oracle TIDAK BISA bilang:
       "coop → jalan pede"
       "non-coop → ngerem keras"
     Ia cuma bisa: "ngerem sedikit lebih sering" (0.735→0.92)

  + Oracle REAKTIF & MYOPIK: putuskan 1 langkah,
    tanpa memikirkan konsekuensi ke depan.
```

### Kenapa POMCP menang?

```
  ORACLE (reaktif)              POMCP (planning)
  ──────────────                ────────────────
  belief ──► tabel ──► aksi     belief
             tetap                │
                                  ▼
                          ┌──────────────────┐
                          │ 100 simulasi UCT  │
                          │ horizon 10 langkah│
                          │ undi tipe, coba   │
                          │ masa depan, pilih │
                          │ yg hasilnya terbaik│
                          └──────────────────┘
                                  │
                                  ▼ aksi (berdasar KONSEKUENSI)
```

```
  ╔═══════════════════════════════════════════════════════╗
  ║  INTI TEMUAN:                                          ║
  ║  Persepsi SEMPURNA + pengendali BODOH (oracle)         ║
  ║         K A L A H                                      ║
  ║  Persepsi tak sempurna + pengendali PINTAR (POMCP)     ║
  ╚═══════════════════════════════════════════════════════╝

  Bahkan POMCP infonya LEBIH JELEK dari oracle
  (belief noisy vs label sempurna) → tapi TETAP MENANG.
  Bukti: yg menentukan bukan "seberapa bagus info",
         tapi "seberapa pintar info dipakai".
```

### ⚠️ Catatan kejujuran (yang supervisor endus)

```
  APPLE-TO-APPLE?   sebagian YA, sebagian TIDAK
  ┌────────────────────────────┬──────────────────────────┐
  │ SAMA (adil)                │ BEDA (perlu di-caveat)    │
  ├────────────────────────────┼──────────────────────────┤
  │ • action space {IDLE,SLOWER}│ • POMCP planning pakai    │
  │ • belief Bayesian sama      │   REWARD INTERNAL beda:   │
  │ • episode identik (seed 17) │     collision = −100      │
  │ • skoring akhir reward sama │     close-call = 1.0       │
  │                            │   (vs reward evaluasi      │
  │                            │    close-call cuma 0.25)   │
  └────────────────────────────┴──────────────────────────┘

  → sebagian keunggulan POMCP BERCAMPUR antara
    "planning lebih pintar" + "objektif lebih anti-tabrak"

  🔬 Eksperimen kontrol: jalankan POMCP dgn reward evaluasi.
     Kalau TETAP menang → klaim "planning>reaksi" anti-peluru.
```

---

<a name="11-temuan"></a>
## 11. Temuan Utama

### Hasil mentah (n=300, episode identik)

```
  Collision rate (makin RENDAH makin baik):

  pomcp  │██▌                    5.0%   ★ terbaik
  oracle │█████▏                10.3%
  belief │███████▌              15.0%
  rule   │█████████▏            18.3%
  random │███████████████████████████████████████████ 87.0%
         └────────────────────────────────────────────
```

### Mana yang signifikan? (ini inti paper)

```
  ┌───────────────────┬──────────────┐
  │ oracle vs rule    │ ✅ SIGNIFIKAN │  → courtesy MEMANG berguna
  │ belief vs rule    │ ❌ TIDAK      │  → padahal filter 98.3% akurat?!
  └───────────────────┴──────────────┘
```

```
  PARADOKS:
  ┌──────────────────────────────────────────────────┐
  │ 1. Oracle (tahu label) menang → courtesy berguna  │
  │ 2. Belief (98.3% akurat) TIDAK menang → KENAPA??   │
  └──────────────────────────────────────────────────┘
```

### Bukan kurang sampel

```
  Diuji ulang n=1.960:  belief 13.8% vs rule 15.1%, p=0.489
  → TETAP tidak signifikan, gap malah MENGECIL
  → ada yg lebih mendasar...
```

### Jawaban: TIMING

```
  Akurasi belief per-langkah (episode non-cooperative):

  langkah 1–5  (AWAL)  : ~71%  ◄── keputusan gap dibuat DI SINI
  langkah 21+  (akhir) : >90%  ◄── baru andal, TAPI SUDAH TELAT

  MEKANISME:
  ┌────────────────────────────────────────────────────┐
  │ Saat ego masuk window, mobil non-cooperative SEMPAT │
  │ MELAMBAT sesaat (respons deteksi ego)               │
  │   → gerakannya MIRIP cooperative                     │
  │   → belief salah tebak di awal                       │
  │   → keputusan kritis diambil saat belief masih buta │
  └────────────────────────────────────────────────────┘
```

---

<a name="12-novelty"></a>
## 12. Kenapa Novelty

Novelty BUKAN simulator/POMDP/ide cooperative. Itu semua sudah ada. Novelty = **3 hal digabung**:

```
  (a) BENCHMARK TERKONTROL + LABEL LATEN TERVERIFIKASI
      └─ di dunia nyata tak ada ground-truth sifat sopir
         di sini disuntik sendiri → label PASTI benar
         → bisa ukur akurasi inferensi secara bersih

  (b) MATCHED-COUNTERFACTUAL EVALUATION
      └─ semua policy diuji di episode IDENTIK (seed sama)
         → kalau A>B, bukan karena A kebetulan dpt skenario mudah
         → uji statistik berpasangan jadi valid

  (c) TEMUAN DIAGNOSTIK yg MELAWAN ASUMSI UMUM
      └─ "akurasi inferensi tinggi ≠ closed-loop lebih aman,
          kalau inferensi andal datang SETELAH keputusan kritis"
```

```
  ╔═══════════════════════════════════════════════════════╗
  ║  PESAN UTAMA (satu kalimat):                           ║
  ║                                                        ║
  ║  Tahu niat sopir lain itu berguna. Tapi kalau kamu     ║
  ║  baru bisa menebaknya dengan benar SETELAH harus       ║
  ║  ambil keputusan, tebakan itu jadi nyaris percuma.     ║
  ╚═══════════════════════════════════════════════════════╝
```

**Pelajaran metodologis:** benchmark intent-inference jangan cuma ukur *"bisakah ditebak?"* tapi juga *"datangnya cukup awal?"* dan *"policy-nya cukup pintar memakainya?"*

### Yang TIDAK diklaim (jujur)
```
  ✗ bukan simulator pertama
  ✗ bukan validasi mobil otonom nyata
  ✗ bukan model psikologi manusia
  ✗ POMDP tidak diklaim terpecahkan optimal
  ✓ benchmark diagnostik di lingkungan terkontrol — itu cukup
```

---

<a name="13-aset-visual"></a>
## 13. Peta Aset Visual (GIF & figur yang sudah ada)

Repo sudah punya visualisasi nyata yang bisa kamu pakai untuk presentasi:

### GIF perbandingan (folder `gifs/`)
Tiap GIF = 1 skenario merge, 1 policy, 1 tipe courtesy:

```
              Cooperative                Non-cooperative
  POMCP    pomcp_cooperative.gif      pomcp_non_cooperative.gif
  Belief   belief_cooperative.gif     belief_non_cooperative.gif
  Oracle   oracle_cooperative.gif     oracle_non_cooperative.gif
  Rule     rule_cooperative.gif       rule_non_cooperative.gif
  Random   random_cooperative.gif     random_non_cooperative.gif
```

### Figur hasil (folder `eval_final_v6_cleanobs/figures/`)

| File | Isi |
|---|---|
| `collision_rate.png` | bar chart collision 5 policy (visual §11) |
| `mean_reward.png` | bar chart reward |
| `belief_convergence.png` | **kurva akurasi belief per-langkah** (visual §6 — failure timing) |
| `mean_min_ttc.png` | time-to-collision minimum |
| `success_rate.png` | tingkat keberhasilan |
| `close_call_rate.png` | tingkat nyaris-tabrak |

> **Untuk presentasi:** `belief_convergence.png` adalah figur paling penting — ia menunjukkan langsung kenapa belief gagal (akurat telat). Pasangkan dengan `collision_rate.png` untuk cerita lengkap.

### Visualizer interaktif
```bash
python sim_visualizer.py
# P=POMCP  B=Belief  O=Oracle  R=Rule  N=Random
# C=paksa cooperative  X=paksa non-cooperative  Z=acak
```

---

## Ringkasan Satu Paragraf

> Kita suntik sifat tersembunyi (ngalah/ngotot) ke mobil merge di highway-env lewat parameter IDM/MOBIL, lalu uji 5 policy di episode identik. Oracle yang tahu label sempurna **kalah** dari POMCP — bukan karena infonya jelek, tapi karena oracle dipaksa pakai tabel aksi reaktif simetris (ngerem 92% di kedua tipe, jadi info sempurnanya tak terpakai), sedangkan POMCP merencanakan ke depan. Temuan utama: filter belief akurat 98.3% offline, tapi **tidak** membuat closed-loop lebih aman, karena di 5 langkah pertama (saat keputusan dibuat) akurasinya cuma 71% — mobil non-cooperative sempat melambat dan terlihat seperti cooperative. **Belief yang benar datang terlambat.**
