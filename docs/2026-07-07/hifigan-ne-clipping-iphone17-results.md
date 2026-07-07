# HiFi-GAN×NE クリップ破綻の別端末再現と速度計測（iPhone 17 / iOS 26.4）

本番12セル（精度3 × 計算デバイス4, shape=fixed262）を新端末 **iPhone 17 / iOS 26.4** で
1スイープ実行し、**(1) Neural Engine 経路で HiFi-GAN 出力が膨張してクリップする破綻**が
別端末でも再現するか、**(2) 各セルの推論速度**を計測した。
[feasibility study](../2026-06-08/interactive-feasibility-direction.md) の「壊れない」「速い」の
2軸を、既知端末とは別の個体で埋めるのが目的。

- **端末**: iPhone 17 / iOS 26.4（従来計測の個体とは別。NE 破綻が端末固有でないかの確認）
- **フレームワーク**: CoreML（`.mlmodelc`, fixed262）。Core AI (.aimodel) は iOS 27 必須のため本計測は対象外
- **日付**: 2026-07-07
- **回数**: n=1（allCombinations 一括スイープ）。速度は 1st-call を含むので暖機スパイクに注意
- **入力**: `input_sample.wav`
- **生データ**: `result_iphone17_26.4_20260707/`（debug snapshot npy / summary.txt / timing json / output wav）※ 生データはコミットしない慣例（本 md が正）
- **真値の取り方**: 音質指標は `waveform_predeemph`（HiFi-GAN の生出力 rms）を主に見る。
  wav ではなく debug snapshot npy 由来（[output_*.wav と debug snapshot npy は別物](../../CLAUDE.md) の運用）

## なぜ predeemph rms を主指標にするか

`waveform_predeemph` は HiFi-GAN が吐いた直後の波形で、de-emphasis フィルタや正規化が
乗る前。ここで振幅が膨らんでいれば「ボコーダ段そのものが壊れている」と言い切れる。
`postdeemph` は de-emphasis で更に増幅されるため破綻がより派手に出るが、原因段の切り分けには
predeemph の方が素直。正常なら predeemph rms ≈ 0.028 に収まる。

## 結果1: クリップ破綻（音質）

| 精度 | デバイス | NE 実行 | predeemph rms | 正常比 | postdeemph min/max | 判定 |
|------|----------|:------:|-------------:|:-----:|-------------------|:----:|
| Float32 | cpuOnly   | – | 0.0293 | ×1.0 | -0.45 / 0.65 | ◎ |
| Float32 | cpuAndGPU | – | 0.0315 | ×1.1 | -0.82 / 0.94 | ◎ |
| Float32 | cpuAndNE  | – ※1 | 0.0271 | ×1.0 | -0.51 / 0.65 | ◎ |
| Float32 | all       | – ※1 | 0.0276 | ×1.0 | -0.92 / 1.10 ※2 | ◯ |
| Float16 | cpuOnly   | – | 0.0278 | ×1.0 | -0.76 / 0.73 | ◎ |
| Float16 | cpuAndGPU | – | 0.0267 | ×1.0 | -0.73 / 0.75 | ◎ |
| **Float16** | **cpuAndNE** | **✓** | **0.1175** | **×4.2** | **-2.09 / 2.08** | **✗ clip** |
| **Float16** | **all**      | **✓** | **0.1175** | **×4.2** | **-2.10 / 2.08** | **✗ clip** |
| Int8 | cpuOnly   | – | 0.0280 | ×1.0 | -0.89 / 0.72 | ◎ |
| Int8 | cpuAndGPU | – | 0.0286 | ×1.0 | -0.84 / 0.78 | ◎ |
| **Int8** | **cpuAndNE** | **✓** | **0.1198** | **×4.3** | **-2.79 / 2.00** | **✗ clip** |
| **Int8** | **all**      | **✓** | **0.1197** | **×4.3** | **-2.79 / 1.99** | **✗ clip** |

※1 Float32 は ANE が fp32 非対応のため、cpuAndNE / all を指定しても NE には乗らず CPU/GPU に落ちる。
だから Float32 は 4 デバイスとも壊れない。
※2 Float32_all の postdeemph max 1.10 は de-emphasis の overshoot。predeemph rms は 0.028 と正常で、
HiFi-GAN 出力自体は壊れていない。

**読み取り**: NE に実際に dispatch される 4 セル（Float16 / Int8 × {cpuAndNE, all}）だけが、
HiFi-GAN 生出力 rms を正常の **約4.2〜4.3倍** に膨らませ、postdeemph で振幅 ±2〜2.8 まで
振り切ってハードクリップする。従来端末で観測した「rms 約4.25倍」（[ANE破綻の段階切り分け](../2026-06-21/audio-quality-mcd-results.md)）が
**別個体 iPhone 17 でもほぼ同倍率で再現**。NE クリップは端末固有の不良ではなく、
HiFi-GAN×NE 経路の構造的な破綻であることが補強された。

## 結果2: 推論速度（1スイープ, 1st-call 含む）

RTF = realTimeFactor（`totalPredictMs / outputDurationMs`）。**< 1.0 で実時間より速い**。
出力長 ≈ 3.04s（262 フレーム / 67072 samples）。

| 精度 | デバイス | encoder | decoder計 | /step | hifigan | 合計ms | RTF | 破綻 |
|------|----------|--------:|----------:|------:|--------:|-------:|----:|:----:|
| Float16 | cpuAndNE  |  18.3 |   991 |  3.78 |  56.3 | 1066 | **0.350** | ✗ |
| Int8    | cpuAndNE  |  18.2 |  1005 |  3.84 |  55.2 | 1078 | **0.354** | ✗ |
| Float16 | all       |  19.2 |   982 |  3.75 |  92.0 | 1094 | **0.359** | ✗ |
| Int8    | all       |  16.7 |  1102 |  4.20 |  79.3 | 1198 | **0.394** | ✗ |
| Float16 | cpuOnly   |   5.8 |  1869 |  7.13 |  76.5 | 1951 | 0.641 | ◎ |
| Int8    | cpuOnly   |   5.1 |  1872 |  7.14 |  72.1 | 1949 | 0.641 | ◎ |
| Float16 | cpuAndGPU |  26.0 |  2519 |  9.61 |  60.8 | 2606 | 0.857 | ◎ |
| Float32 | all       |  20.9 |  3278 | 12.51 | 141.1 | 3440 | 1.131 | ◯ |
| Float32 | cpuAndNE  |   9.3 |  3742 | 14.28 | 164.0 | 3915 | 1.287 | ◎ |
| Int8    | cpuAndGPU | 984.5 |  3092 | 11.80 | 572.3 | 4649 | 1.528 ※ | ◎ |
| Float32 | cpuOnly   |  18.9 |  5507 | 21.02 | 154.5 | 5680 | 1.867 | ◎ |
| Float32 | cpuAndGPU |1655.4 |  3953 | 15.09 |1037.4 | 6646 | 2.185 ※ | ◎ |

※ GPU 経路の encoder / hifigan が突出（1655ms, 1037ms 等）しているのは 1st-call のプラン生成・
暖機コスト。2回目以降は落ちるはず（従来の [2回目収束](../../CLAUDE.md) と同種）。n=1 のため要再計測。

## 読み取れること

- **最速の4セルが、そのまま壊れる4セル**。RTF 0.35〜0.39 の上位4つ（Float16 / Int8 × {cpuAndNE, all}）は
  全て NE dispatch セルで、全て HiFi-GAN がクリップ破綻している。「速い」と「壊れない」が
  この端末でも**真正面から衝突**する。feasibility の成立マップ上、NE 系セルは
  「速いが音が壊れる」象限に固定される。
- **壊れず実時間内で回る（RTF<1）のは CPU/GPU 系のうち Float16/Int8 の cpuOnly・cpuAndGPU の4セル**。
  Float16_cpuOnly (RTF 0.64) / Int8_cpuOnly (0.64) が「壊れない×速い」の最有力候補。
- **Float32 は壊れないが遅い**（RTF 1.13〜2.19、GPU は 1st-call で更に悪化）。ANE を使えない分、
  安全側だがインタラクティブには重い。
- decoder が支配的（合計の 8〜9 割）。NE 系が速いのは decoder/step が 3.8ms 前後まで縮むから。
  裏を返すと、その NE 経路が HiFi-GAN 段で振幅を壊している。**速さの源泉と破綻の源泉が同じ NE 経路**。

## 次にやること

- GPU 系セルの 2nd-call 以降を計測し、1st-call 暖機スパイクを分離（n≥2）。
- NE クリップの op 単位の原因は依然未確定（[ANE破綻の段階切り分け](../2026-06-21/audio-quality-mcd-results.md)）。
  Core AI (.aimodel) 版でこの破綻が再現するか／SDK で層単位に切れるかが移植研究の主眼
  （[Core AI移植](../2026-06-25/coreai-device-inference-design.md)）。
