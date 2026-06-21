# ANE 破綻の段階切り分け ── どのステージで音が壊れるか

Float16 / Int8 × Neural Engine 経路で音質が破綻する原因箇所を、合成パイプラインの
**各ステージ境界の中間出力**を突き合わせて特定した調査。

- 関連: [音質（使える音）軸 MCD 計測](audio-quality-mcd-results.md)（ANE 4セルだけ MCD 11.6〜12.0dB）、
  [clipping 観測](../2026-05-19/stability-matrix-analysis.md)、
  [dispatch マップ](../2026-05-19/compute-plan-analysis.md)、
  [Phase2-mini K 実験](../2026-05-19/phase2-mini-rescue-results.md)
- 生成スクリプト: `scripts/extract_pytorch_stages.py` / `scripts/compare_stages.py` /
  `scripts/compare_hifigan_isolation.py` / `scripts/stage_metrics.py`
- 自動生成サブ表: [stage-compare-device.md](stage-compare-device.md) /
  [stage-compare-pytorch.md](stage-compare-pytorch.md) / [hifigan-isolation.md](hifigan-isolation.md)

---

## 目的

PronounSE 合成パイプラインを CoreML で iPhone 実行したとき、Float16 / Int8 × Neural Engine の
条件だけ音が破綻する。これが

- encoder 側
- decoder / Transformer 側
- mel-spectrogram 生成後
- HiFi-GAN vocoder 側
- 最終 wave 出力のみ

のどこで起きているかを、中間出力を段階比較して切り分ける。

## 背景

対 PyTorch baseline の MCD 計測（[audio-quality-mcd-results.md](audio-quality-mcd-results.md)）で、
CPU/GPU 系8セルが 3.9〜6.6 dB に収まる一方、**F16/Int8 × {cpuAndNE, all} の4セルだけ
11.6〜12.0 dB** に隔離された。「速い」「壊れない」軸でも同じ4セルが落ちており、3軸が同じ ANE 4セルを
同時棄却している。本調査はその破綻が**パイプラインのどの段で生まれるか**を確定する。

## 実験条件

パイプラインは `mel → Encoder → Decoder(自己回帰) → postnet → HiFi-GAN → de-emphasis → wave`。
iOS アプリは各 run で 5 つの境界を `.npy` で debug snapshot に書き出している
（`audio/<archive>/debug/<runId>/`）。これを 2 通りで比較した。

| 比較 | reference | 対象 | 性質 |
|---|---|---|---|
| **A. 実機 良 vs 壊れ** | 実機 `f16 cpuAndGPU`（良） | 実機 `f16/int8 × {cpuAndNE, all}`（壊）＋良コントロール | **混入なし**。全 run が `mel_normalized` の sha256 一致＝同一前処理なので、差は computeUnit だけに帰属できる |
| B. PyTorch vs 実機 | PyTorch baseline | 実機各 run | 絶対アンカー。ただし下記の前処理差あり |
| **C. HiFi-GAN 単体（Mac）** | PyTorch HiFi-GAN | CoreML HiFi-GAN 6条件（同一 postnet 入力） | 入力固定で HiFi-GAN だけ取り出し、エンジン別に出力を比較 |

入力は全条件 `input_sample.wav`（iOS バンドルと sha256 一致）。実機データは 2026-05-19 の
phase1 スイープ（`audio/iPhone_3_phase1_20260519/`、各条件 fixed262, T=262）。

> **前処理差の注意（比較 B が純粋な切り分けにならない理由）**: iOS の mel フロントエンド（Swift STFT）は
> PyTorch の `get_spectrograms`（librosa）と**一致しない**。同じ wav で
> フレーム数 262(iOS) vs 266(librosa)、`mel_normalized` の mean 0.42 vs 0.17、max 1.0 vs 0.71、
> cosine 0.92。よって PyTorch vs 実機は最初の mel 段で既に入力がズレており、下流の差の一部が
> ANE ではなく前処理由来になる。**ANE の純粋な切り分けは比較 A（実機 良 vs 壊れ）で行う。**

## 比較した中間出力

| 境界 | npy ファイル名 | 意味 |
|---|---|---|
| mel | `mel_normalized.npy` | Encoder 入力メルスペクトログラム [T,256] |
| encoder | `encoder_output.npy` | Encoder 出力 memory [T,512] |
| postnet | `postnet_output.npy` | Decoder/postnet 出力 = HiFi-GAN 入力 mel [T,256] |
| HiFi-GAN 出力 | `waveform_predeemph.npy` | HiFi-GAN 出力波形（de-emphasis 前）[1,N] |
| 最終波形 | `waveform_postdeemph.npy` | de-emphasis 後の最終波形 [1,N] |

指標: shape / dtype / min / max / mean / std / MAE / RMSE / cosine / peak / RMS / NaN・Inf。
判定は cosine と `rms_ratio = cur_rms / ref_rms` のヒューリスティック（`scripts/stage_metrics.py`）。
**波形は生サンプル cosine が精度・エンジン差で簡単に下がるため振幅 `rms_ratio` を主指標**にし、
tensor 段は cosine を主にした。

## 結果表

### A. 実機 良(f16 cpuAndGPU) vs 各 run ── 混入なしの切り分け（主結果）

| stage | condition | cosine | rms_ratio | min | max | NaN/Inf | 判定 |
|---|---|---:|---:|---:|---:|:---:|:---:|
| mel | f32 cpuOnly（良） | 1.0000 | 1.000 | 1e-08 | 1.0 | なし | 🟢 正常 |
| encoder | f32 cpuOnly（良） | 1.0000 | 1.000 | -5.15 | 9.78 | なし | 🟢 正常 |
| postnet | f32 cpuOnly（良） | 1.0000 | 1.000 | -0.057 | 0.731 | なし | 🟢 正常 |
| HiFi-GAN出力 | f32 cpuOnly（良） | 0.9779 | 1.001 | -0.45 | 0.36 | なし | 🟢 正常 |
| 最終波形 | f32 cpuOnly（良） | 0.9904 | 1.007 | -0.73 | 0.74 | なし | 🟢 正常 |
| mel | int8 cpuAndGPU（良） | 1.0000 | 1.000 | 1e-08 | 1.0 | なし | 🟢 正常 |
| encoder | int8 cpuAndGPU（良） | 0.9995 | 1.001 | -5.15 | 9.78 | なし | 🟢 正常 |
| postnet | int8 cpuAndGPU（良） | 0.9909 | 0.986 | -0.07 | 0.73 | なし | 🟢 正常 |
| HiFi-GAN出力 | int8 cpuAndGPU（良） | 0.1729 | 1.057 | -0.47 | 0.37 | なし | 🟡 軽微な差 |
| 最終波形 | int8 cpuAndGPU（良） | 0.2671 | 1.108 | -0.78 | 0.80 | なし | 🟡 軽微な差 |
| mel | **f16 cpuAndNE（壊）** | 1.0000 | 1.000 | 1e-08 | 1.0 | なし | 🟢 正常 |
| encoder | **f16 cpuAndNE（壊）** | 1.0000 | 1.000 | -5.15 | 9.77 | なし | 🟢 正常 |
| postnet | **f16 cpuAndNE（壊）** | 1.0000 | 0.999 | -0.066 | 0.736 | なし | 🟢 正常 |
| **HiFi-GAN出力** | **f16 cpuAndNE（壊）** | **0.0173** | **4.254** | -0.94 | 0.96 | なし | **🔴 破綻** |
| **最終波形** | **f16 cpuAndNE（壊）** | 0.1822 | 2.662 | -2.18 | 2.11 | なし | **🔴 破綻** |
| mel | **f16 all（壊）** | 1.0000 | 1.000 | 1e-08 | 1.0 | なし | 🟢 正常 |
| encoder | **f16 all（壊）** | 1.0000 | 1.000 | -5.15 | 9.77 | なし | 🟢 正常 |
| postnet | **f16 all（壊）** | 1.0000 | 0.999 | -0.066 | 0.735 | なし | 🟢 正常 |
| **HiFi-GAN出力** | **f16 all（壊）** | **0.0197** | **4.249** | -0.94 | 0.96 | なし | **🔴 破綻** |
| **最終波形** | **f16 all（壊）** | 0.1804 | 2.660 | -2.18 | 2.11 | なし | **🔴 破綻** |
| mel | **int8 cpuAndNE（壊）** | 1.0000 | 1.000 | 1e-08 | 1.0 | なし | 🟢 正常 |
| encoder | **int8 cpuAndNE（壊）** | 0.9995 | 1.000 | -5.15 | 9.78 | なし | 🟢 正常 |
| postnet | **int8 cpuAndNE（壊）** | 0.9913 | 0.985 | -0.07 | 0.73 | なし | 🟢 正常 |
| **HiFi-GAN出力** | **int8 cpuAndNE（壊）** | **0.0024** | **4.367** | -0.96 | 0.97 | なし | **🔴 破綻** |
| **最終波形** | **int8 cpuAndNE（壊）** | 0.0489 | 2.835 | -2.3 | 2.3 | なし | **🔴 破綻** |
| mel | **int8 all（壊）** | 1.0000 | 1.000 | 1e-08 | 1.0 | なし | 🟢 正常 |
| encoder | **int8 all（壊）** | 0.9995 | 1.000 | -5.15 | 9.78 | なし | 🟢 正常 |
| postnet | **int8 all（壊）** | 0.9913 | 0.985 | -0.07 | 0.73 | なし | 🟢 正常 |
| **HiFi-GAN出力** | **int8 all（壊）** | **0.0024** | **4.360** | -0.96 | 0.97 | なし | **🔴 破綻** |
| **最終波形** | **int8 all（壊）** | 0.0469 | 2.836 | -2.3 | 2.3 | なし | **🔴 破綻** |

完全な指標（MAE/RMSE/std/peak 等）は [stage-compare-device.md](stage-compare-device.md) を参照。

### C. HiFi-GAN 単体（Mac, 同一 postnet 入力）── 能動的な再現

基準 = PyTorch HiFi-GAN（出力 rms=0.02689, peak=0.4584）。

| condition | cosine | rms_ratio | peak | 判定 |
|---|---:|---:|---:|:---:|
| CoreML F32 cpuOnly | 1.0000 | 1.000 | 0.458 | 🟢 正常 |
| CoreML F16 cpuAndGPU | 1.0000 | 0.998 | 0.458 | 🟢 正常 |
| **CoreML F16 cpuAndNE** | **0.0067** | **4.360** | 0.969 | **🔴 破綻** |
| CoreML Int8 cpuAndGPU | 0.9979 | 0.998 | 0.464 | 🟢 正常 |
| **CoreML Int8 cpuAndNE** | **0.0067** | **4.370** | 0.971 | **🔴 破綻** |

完全な指標は [hifigan-isolation.md](hifigan-isolation.md)、聴感用 wav は `docs/2026-06-21/hifigan_wav/`（git 管理外）。

> **Mac の cpuAndNE が iPhone ANE の破綻を再現した**: 実機 ANE 壊れ run の HiFi-GAN 出力 rms は
> PyTorch 比 **4.24×**（0.11403 / 0.02689）。Mac ローカル F16 cpuAndNE は **4.36×**でほぼ一致。
> → この破綻は「iPhone 固有」ではなく、**この HiFi-GAN を NE で F16/Int8 実行すること自体に内在**する。
> （一般には Mac の ANE 配置 ≠ iPhone とは限らないが、本モデルの HiFi-GAN では再現した）

## どの段階で破綻したか

**HiFi-GAN vocoder を Neural Engine で実行する段で破綻する。** 根拠:

1. **HiFi-GAN 入力までは全て正常（比較 A）**。壊れ4セルの `mel` / `encoder` / `postnet` はいずれも
   良 run と cosine ≈ 1.0、rms_ratio ≈ 1.0。**HiFi-GAN の入力 mel は良 run とほぼビット一致**。
2. **HiFi-GAN 出力で初めて破綻**。同じ入力なのに出力波形の rms が **約 4.25〜4.37 倍**に膨張し、
   波形 cosine が ≈ 0（無相関）。peak が 0.94〜0.97 に達し、de-emphasis 後は |振幅| が 2 を超えて
   clipping（最終波形 min/max ≈ ±2.1〜2.3）。
3. **入力を固定した HiFi-GAN 単体実験（比較 C）でも同じ**。同一 postnet を入れて F32/F16-GPU/Int8-GPU は
   rms_ratio ≈ 1.0（正常）、F16/Int8 × cpuAndNE だけ rms_ratio ≈ 4.36（破綻）。
   → encoder/decoder を介さず HiFi-GAN だけで破綻が再現＝**原因は HiFi-GAN × NE に限局**。

つまり encoder 側・decoder 側・mel 生成・最終 wave のいずれでもなく、**HiFi-GAN vocoder 段**。
これは Phase4 dispatch 観測（HiFi-GAN が NE に乗る構成で振幅破綻）、Phase2-mini K 実験
（postnet は cpuAndNE 単独と bit-identical のまま、HiFi-GAN だけ cpuAndGPU 退避で clipping が消える）と整合する。

> 補足: 比較 A で良コントロールの `int8 cpuAndGPU` が HiFi-GAN 出力 cosine 0.17 と低く出るが
> rms_ratio は 1.06（振幅正常）で、MCD 上も良セル（5.93dB）。これは**生サンプル cosine が精度差で
> 簡単に下がる**ことの実例で、破綻判定に cosine 単独を使えないこと（rms_ratio を主にした理由）を裏づける。

## 今後の確認事項

- **op 単位の特定は未確定**。「HiFi-GAN × NE で振幅が約 4 倍」までは確定したが、Generator 内の
  どの層（ConvTranspose の upsample / ResBlock / 最終 tanh 直前）で膨らみ始めるかは未特定。
  `scripts/generate_hifigan_reference.py` が per-block 中間 tensor を出せるので、NE 実行時の
  block 出力を取れれば層単位まで追える（多出力 CoreML モデルが必要）。
- **約 4 倍という比率の意味**。tanh 前後の飽和・正規化スケールとの関係（なぜ ~4×で頭打ちか）は要検討。
- **iOS mel フロントエンド差（262 vs 266, cosine 0.92）**は本破綻とは別問題だが、MCD の床（~4dB）に
  効いている可能性。Swift STFT と librosa の差分は別途の課題。
- **救済策の効果測定**: 出力正規化（J）・HiFi-GAN だけ cpuAndGPU 退避（K）を本切り分けの指標
  （rms_ratio）で定量化すると、phase2-mini の聴感所見を数値で裏づけられる。

## 発表用の短い結論

> ANE 経路の音質破綻は、encoder でも decoder でもなく **HiFi-GAN vocoder を Neural Engine で
> F16/Int8 実行する段**で起きる。HiFi-GAN の入力 mel は良条件とほぼビット一致なのに、出力波形の振幅だけが
> **約 4 倍に膨張して clipping** する。同じ mel を入れた HiFi-GAN 単体実験（Mac）でも cpuAndNE だけ
> 4.36 倍になり、実機の 4.24 倍と一致した。破綻は HiFi-GAN × NE に限局しており、HiFi-GAN を
> CPU/GPU に退避すれば回避できる、というのが切り分けの結論。

## 再現コマンド

```bash
# 0. PyTorch baseline の各ステージ中間出力（真値リファレンス）を生成
cd PronounSE && venv/bin/python ../scripts/extract_pytorch_stages.py \
    --input input_sample.wav \
    --out-dir ../data/2026-06-21/pytorch_reference/input_sample
cd ..

B=audio/iPhone_3_phase1_20260519/debug

# A. 実機 良 vs 壊れ（混入なしの切り分け, 主結果）
PronounSE/venv/bin/python scripts/compare_stages.py \
    --reference "$B/20260519_212806_f16GpuRepeat2_Float16_cpuAndGPU" \
    --run "$B/20260519_213430_f32CpuRepeat1_Float32_cpuOnly" \
    --run "$B/20260519_213036_int8GpuRepeat1_Int8_cpuAndGPU" \
    --run "$B/20260519_215227_f16NeRepeat1_Float16_cpuAndNE" \
    --run "$B/20260519_215702_f16AllRepeat1_Float16_all" \
    --run "$B/20260519_220617_int8NeRepeat1_Int8_cpuAndNE" \
    --run "$B/20260519_221054_int8AllRepeat1_Int8_all" \
    --md docs/2026-06-21/stage-compare-device.md --plots

# B. PyTorch vs 実機（絶対アンカー, 前処理差に注意）
PronounSE/venv/bin/python scripts/compare_stages.py \
    --reference data/2026-06-21/pytorch_reference/input_sample \
    --run "$B/20260519_212806_f16GpuRepeat2_Float16_cpuAndGPU" \
    --run "$B/20260519_215227_f16NeRepeat1_Float16_cpuAndNE" \
    --run "$B/20260519_220617_int8NeRepeat1_Int8_cpuAndNE" \
    --md docs/2026-06-21/stage-compare-pytorch.md

# C. HiFi-GAN 単体（Mac, 同一 postnet 入力）
PronounSE/venv/bin/python scripts/compare_hifigan_isolation.py \
    --postnet "$B/20260519_212806_f16GpuRepeat2_Float16_cpuAndGPU/postnet_output.npy" \
    --device-broken "$B/20260519_215227_f16NeRepeat1_Float16_cpuAndNE" \
    --md docs/2026-06-21/hifigan-isolation.md --save-wav
```
