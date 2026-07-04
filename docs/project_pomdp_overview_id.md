# Penjelasan Sederhana HiddenCourtesyMerge-Sim dan Model POMDP

## Inti Project Ini

Project ini memakai simulator `highway-env merge-v0`, yaitu simulator jalan tol
dengan skenario kendaraan dari ramp yang ingin masuk ke jalan utama.

Simulator `merge-v0` sebenarnya sudah menyediakan:

- jalan tol,
- ramp merge,
- kendaraan ego,
- kendaraan lain,
- dinamika kendaraan,
- aksi seperti jalan terus atau melambat.

Jadi novelty project ini **bukan** membuat simulator merge dari nol.

Yang project ini tambahkan adalah:

> Ada satu sifat pengemudi merge yang disembunyikan dari ego vehicle:
> apakah kendaraan merge itu cooperative atau non-cooperative.

Artinya:

- `cooperative`: kendaraan merge lebih mau memberi ruang atau melambat.
- `non-cooperative`: kendaraan merge lebih agresif, menjaga speed tinggi, dan
  tidak banyak memberi ruang.

Ego vehicle tidak diberi tahu label ini. Ego hanya bisa menebak dari gerakan
kendaraan merge.

Pertanyaan utama project ini:

> Kalau ego bisa menebak apakah kendaraan merge cooperative atau
> non-cooperative, apakah keputusan ego jadi lebih aman?

Temuan utama:

> Belief atau tebakan ego bisa akurat kalau dilihat secara offline, tetapi tetap
> belum tentu membuat closed-loop driving lebih aman, karena tebakan yang benar
> sering datang terlambat.

Analogi sederhana:

Bayangkan kamu sedang mau masuk atau melewati area merge. Ada mobil dari ramp.
Kamu ingin tahu apakah dia akan memberi jalan atau memaksa masuk.

Kalau kamu menunggu cukup lama, kamu mungkin bisa tahu dia tipe yang mana.
Masalahnya, keputusan untuk melambat atau tetap jalan harus dibuat lebih awal.
Jadi prediksi yang benar tetapi terlambat tidak banyak membantu keselamatan.

## Project Ini Sebenarnya Menguji Apa?

Project ini menguji hubungan antara tiga hal:

1. **Inference**
   - Apakah ego bisa menebak hidden courtesy?

2. **Belief-based decision**
   - Kalau ego punya belief, apakah policy jadi lebih baik?

3. **Closed-loop safety**
   - Saat policy benar-benar dijalankan di simulator, apakah collision dan
     near-collision berkurang?

Hasil menariknya:

> Akurasi belief yang tinggi tidak otomatis berarti safety meningkat.

Itulah failure mode utama yang ditemukan.

## Kenapa Perlu POMDP?

POMDP dipakai karena ada bagian penting dari dunia yang tidak bisa dilihat
langsung oleh ego.

Dalam kasus ini, bagian tersembunyi itu adalah:

```text
m = courtesy type
```

Ego tidak tahu apakah:

```text
m = cooperative
```

atau:

```text
m = non_cooperative
```

Tanpa POMDP, ego hanya melihat data fisik:

```text
jarak, kecepatan, akselerasi
```

Dengan POMDP, ego juga menyimpan belief:

```text
P(cooperative) = 0.7
P(non_cooperative) = 0.3
```

Jadi POMDP membantu menjawab:

> Dengan informasi yang belum pasti ini, action apa yang paling masuk akal?

## Model POMDP Kita

Model POMDP ditulis sebagai:

```text
M = (S, A, O, T, Z, R, gamma, b0)
```

Penjelasan sederhananya:

| Komponen | Arti sederhana |
|---|---|
| `S` | state atau kondisi dunia |
| `A` | action yang bisa dipilih ego |
| `O` | observation yang dilihat ego |
| `T` | aturan perubahan state |
| `Z` | model likelihood observation |
| `R` | reward atau nilai evaluasi |
| `gamma` | discount factor untuk planning |
| `b0` | belief awal |

## State Space

Ada dua jenis state yang perlu dibedakan.

### 1. Full Simulator State

Ini adalah state lengkap yang dipakai `highway-env` di dalam simulator.

Isinya bisa mencakup:

- posisi semua kendaraan,
- kecepatan semua kendaraan,
- lane setiap kendaraan,
- geometri jalan,
- parameter IDM/MOBIL,
- traffic lain,
- detail internal simulator.

Kita bisa tulis:

```text
x_t^env = seluruh state internal highway-env
```

State ini dipakai simulator untuk menjalankan dunia.

### 2. Compact POMDP State

Untuk paper dan policy, kita pakai state yang lebih ringkas:

```text
s_t = (x_e, v_e, x_m, v_m, d_t, Delta v_t, m)
```

Artinya:

| Simbol | Arti |
|---|---|
| `x_e` | posisi longitudinal ego |
| `v_e` | kecepatan ego |
| `x_m` | posisi longitudinal kendaraan merge |
| `v_m` | kecepatan kendaraan merge |
| `d_t = x_m - x_e` | jarak relatif antara merge vehicle dan ego |
| `Delta v_t = v_m - v_e` | kecepatan relatif |
| `m` | hidden courtesy type |

State space-nya:

```text
S = X_e x V_e x X_m x V_m x D x DeltaV x M
```

dengan:

```text
M = {cooperative, non_cooperative}
```

Intinya:

- bagian fisik seperti posisi dan speed bisa diamati,
- courtesy type `m` tidak bisa diamati langsung.

## Hidden State

Hidden state kita adalah:

```text
m in {cooperative, non_cooperative}
```

`m` dipilih sekali saat episode dimulai.

Selama satu episode, `m` tidak berubah:

```text
P(m_{t+1} = m_t) = 1
```

Ini bukan klaim bahwa manusia asli tidak pernah berubah niat.

Ini adalah pilihan desain benchmark supaya kita bisa fokus pada pertanyaan:

> Kalau hidden driver type fixed tetapi tidak terlihat, apakah ego bisa
> menebaknya dan memakai tebakan itu untuk driving?

## Action Space

Untuk policy terstruktur, action yang dipakai adalah:

```text
A = {IDLE, SLOWER}
```

Artinya:

| Action | Arti |
|---|---|
| `IDLE` | ego mempertahankan perilaku saat ini |
| `SLOWER` | ego melambat untuk membuat ruang |

Kenapa hanya dua?

Karena kita ingin membandingkan rule, belief, dan oracle secara fair. Kalau
semua policy punya action table yang sama, maka perbedaannya lebih jelas:

> Apakah informasi courtesy membantu atau tidak?

Random policy boleh memakai action lebih luas sebagai baseline bawah, tetapi
policy utama dibuat terkontrol.

### Kenapa Tidak Memakai Semua Action Bawaan `highway-env`?

Sebenarnya project ini tetap memakai action bawaan `highway-env`. Kita tidak
membuat action baru. Yang kita lakukan adalah memakai subset dari meta-action
bawaan.

Action bawaan `highway-env` mencakup:

```text
LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER
```

Tetapi policy utama yang terstruktur hanya memakai:

```text
IDLE, SLOWER
```

Alasannya bukan karena action lain tidak ada, tetapi karena kita ingin menjaga
eksperimen tetap terkontrol.

Pertanyaan utama project ini adalah:

> Apakah hidden courtesy information membantu ego memutuskan kapan harus
> melambat atau memberi ruang?

Kalau semua action langsung dipakai, hasil eksperimen menjadi lebih sulit
dibaca. Misalnya suatu policy lebih aman. Penyebabnya bisa jadi:

- belief-nya lebih baik,
- action table-nya lebih baik,
- policy bisa pindah lane,
- policy bisa accelerate,
- atau policy punya action space yang lebih ekspresif.

Dengan membatasi policy utama ke `IDLE` dan `SLOWER`, kita membuat perbandingan
lebih fair:

```text
rule_policy   : action table sama, tidak tahu courtesy
belief_policy : action table sama, memakai belief courtesy
oracle_policy : action table sama, tahu courtesy asli
```

Jadi perbedaan hasil lebih mudah ditafsirkan sebagai efek informasi courtesy,
bukan efek action tambahan.

Analogi sederhana:

Kalau kita ingin menguji apakah seseorang mengambil keputusan lebih baik karena
punya informasi tambahan, semua peserta harus diberi pilihan aksi yang sama.
Kalau satu peserta boleh pindah lane dan peserta lain tidak, maka kita tidak
tahu apakah dia menang karena informasinya lebih baik atau karena pilihannya
lebih banyak.

Dalam `merge-v0`, konflik utama yang ingin diuji adalah:

```text
ego tetap jalan atau ego melambat untuk memberi gap?
```

Itulah kenapa `IDLE` dan `SLOWER` cukup untuk eksperimen utama.

Namun ini juga menjadi limitation:

> Project ini belum menguji apakah action seperti lane change atau acceleration
> akan menghasilkan policy yang lebih baik di skenario merge yang lebih luas.

Kalimat paper yang aman:

> We use highway-env's native discrete meta-actions, but restrict the structured
> non-random policies to `{IDLE, SLOWER}` to isolate the effect of hidden
> courtesy information from action-space expressiveness.

## Observation

Ego tidak melihat hidden label `m`.

Ego hanya melihat motion cues:

```text
o_t = (ard_t, mvs_t, mva_t)
```

Artinya:

| Feature | Arti |
|---|---|
| `ard` | absolute relative distance |
| `mvs` | merge vehicle speed |
| `mva` | merge vehicle acceleration |

Observation hanya dipakai saat kendaraan berada di interaction window:

```text
2 m < d_t < 40 m
```

Di luar window ini, belum ada informasi courtesy yang cukup bermakna.

## Observation Model

Observation model menjawab:

> Kalau driver itu cooperative, seberapa mungkin kita melihat observation ini?
> Kalau driver itu non-cooperative, seberapa mungkin kita melihat observation ini?

Model likelihood yang dipakai:

```text
Z(o_t | m) = product_i N(o_{t,i}; mu_{m,i}, sigma_{m,i}^2)
```

Dalam bahasa sederhana:

- untuk setiap tipe driver, kita pelajari pola motion cues,
- lalu kita hitung observation sekarang lebih cocok ke tipe yang mana.

Contoh intuisi:

- Kalau merge vehicle melambat dan membuka gap, observation lebih cocok dengan
  cooperative.
- Kalau merge vehicle menjaga speed tinggi dan tidak memberi ruang, observation
  lebih cocok dengan non-cooperative.

## Belief Update

Belief adalah tebakan probabilistik ego terhadap courtesy type.

Contoh:

```text
b_t(cooperative) = 0.6
b_t(non_cooperative) = 0.4
```

Artinya ego percaya 60% bahwa kendaraan merge cooperative dan 40% bahwa dia
non-cooperative.

Belief awal:

```text
b0 = [0.5, 0.5]
```

Artinya sebelum melihat evidence, ego netral.

### General POMDP Belief Update

Secara umum:

```text
b_{t+1}(m') =
  eta * Z(o_{t+1} | m') * sum_m P(m' | m) b_t(m)
```

Artinya:

1. Ambil belief sebelumnya.
2. Prediksi kemungkinan hidden state berikutnya.
3. Lihat observation baru.
4. Naikkan belief untuk tipe yang lebih cocok dengan observation.
5. Normalize supaya total probabilitas = 1.

### Karena Courtesy Fixed

Karena dalam benchmark ini courtesy tidak berubah:

```text
P(m' | m) = 1 jika m' = m
P(m' | m) = 0 jika m' != m
```

Maka update menjadi lebih sederhana:

```text
tilde_b_{t+1}(m) =
  eta * Z(o_{t+1} | m) * b_t(m)
```

Lalu kita pakai shrinkage:

```text
b_{t+1} =
  (1 - lambda) * tilde_b_{t+1} + lambda * b0
```

dengan:

```text
lambda = 0.08
```

Shrinkage ini dipakai supaya belief tidak terlalu cepat menjadi overconfident,
karena observation antar timestep saling berkorelasi.

## Transition Model

Transition model menjawab:

> Kalau sekarang state-nya seperti ini dan ego memilih action tertentu, state
> berikutnya akan menjadi apa?

Dalam project ini transition dibagi dua.

### 1. Transition Hidden Courtesy

Courtesy tidak berubah:

```text
P(m' | m) = 1 jika m' = m
```

Jadi:

```text
cooperative tetap cooperative
non_cooperative tetap non_cooperative
```

### 2. Transition Observable Traffic

Bagian fisik dunia dijalankan oleh simulator:

```text
s'_obs ~ T_env(s_obs, a, m, xi)
```

Artinya state berikutnya tergantung pada:

| Komponen | Arti |
|---|---|
| `s_obs` | state fisik yang terlihat |
| `a` | action ego |
| `m` | hidden courtesy type |
| `xi` | randomness dan detail simulator lain |

Gabungan transition-nya:

```text
T(s' | s, a) =
  delta(m' = m) * T_env(s'_obs | s_obs, a, m)
```

Dalam bahasa sederhana:

- hidden courtesy tetap sama,
- posisi/speed kendaraan berubah mengikuti simulator,
- perilaku merge vehicle dipengaruhi oleh parameter courtesy.

Contoh:

```text
Jika ego memilih SLOWER:
  ego melambat dan gap mungkin membesar.

Jika merge vehicle cooperative:
  dia lebih cenderung menciptakan/menerima gap aman.

Jika merge vehicle non-cooperative:
  dia lebih cenderung menjaga speed tinggi dan gap kecil.
```

## Reward

Reward evaluasi:

```text
R_t = r_env_t - 0.25 * 1[TTC_t < 3s atau d_t in (0, 15m)]
```

Artinya:

- ego mendapat reward dari simulator,
- tetapi mendapat penalti kalau terlalu dekat atau TTC terlalu kecil.

Reward ini dipakai untuk mengukur tradeoff antara:

- safety,
- throughput,
- near-collision risk.

## Policy yang Dibandingkan

| Policy | Arti |
|---|---|
| `random_policy` | baseline bawah, action acak |
| `rule_policy` | tidak memakai belief, selalu prior 50/50 |
| `belief_policy` | memakai Bayesian belief |
| `oracle_policy` | tahu hidden courtesy asli, tapi masih memakai heuristic action table |
| `pomcp_policy` | memakai online planning dengan belief |

Catatan penting:

`oracle_policy` bukan oracle optimal. Dia hanya oracle-heuristic.

Artinya:

> Dia tahu label courtesy asli, tetapi masih terbatas oleh action table
> heuristic.

## Apa yang Ditemukan?

Temuan utama:

1. Belief model bisa akurat untuk klasifikasi courtesy.
2. Tetapi `belief_policy` tidak signifikan lebih aman daripada `rule_policy`.
3. Penyebabnya belief paling lemah pada awal interaction window.
4. Padahal keputusan safety-critical harus dibuat di awal window.
5. POMCP lebih baik karena planning bisa mencari action sequence, bukan hanya
   mengikuti action table sederhana.

Kalimat sederhana:

> Project ini menunjukkan bahwa tahu niat driver lain itu berguna, tetapi
> menebaknya terlalu lambat bisa membuat belief tidak membantu keputusan
> keselamatan.

## Scientific Novelty

Novelty yang aman diklaim:

> HiddenCourtesyMerge-Sim adalah benchmark terkontrol untuk menguji apakah
> inference hidden courtesy benar-benar membantu closed-loop merge safety.

Dan finding utamanya:

> Accurate offline belief estimation can fail to improve closed-loop safety
> when the belief becomes reliable only after the safety-critical decision point.

Versi Bahasa Indonesia:

> Estimasi niat driver yang akurat secara offline belum tentu membuat driving
> lebih aman, kalau estimasi itu baru akurat setelah momen keputusan penting
> sudah lewat.

## Apa yang Tidak Diklaim

Project ini tidak mengklaim:

- pertama kali membuat merge simulator,
- pertama kali memakai POMDP untuk merge,
- pertama kali memakai cooperative/non-cooperative driver,
- validasi real-world autonomous driving,
- model psikologi manusia,
- POMCP mengalahkan oracle optimal.

Klaim yang benar:

> Project ini menyediakan controlled benchmark dan diagnostic evidence untuk
> memahami kapan hidden-courtesy belief membantu, kapan gagal, dan kenapa.
