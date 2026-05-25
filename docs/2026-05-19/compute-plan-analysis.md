# Phase 4 MLComputePlan 観測結果（2026-05-19 実機 iPhone 13）

生データ: [`mlcomputeplan-dispatch-map.md`](mlcomputeplan-dispatch-map.md), `audio/iPhone_3_phase4_20260519_round2/*.json`

18 plans 取得後の整理。観測事実と推論（候補）を分けて記述する。

## 1. dispatch 観測（実機計測）

iOS 17+ の `MLComputePlan` API から取得した per-op device assignment。

### 1.1 Decoder

| precision | computeUnit 設定 | CPU | GPU | NE |
|---|---|---:|---:|---:|
| F32 | cpuOnly | 195 | 0 | 0 |
| F32 | cpuAndGPU | 0 | 195 | 0 |
| F32 | cpuAndNE | 195 | 0 | 0 |
| F32 | all | 0 | 195 | 0 |
| F16 | cpuAndGPU | 0 | 204 | 0 |
| F16 | cpuAndNE | 16 | 0 | 188 |
| F16 | all | 16 | 0 | 188 |
| Int8 | cpuAndGPU | 0 | 208 | 0 |
| Int8 | cpuAndNE | 20 | 0 | 188 |
| Int8 | all | 20 | 0 | 188 |

事実:
- F32 は computeUnits に NE を含めても NE に dispatch されていない（F32 は ANE 非対応）
- F16 / Int8 × {cpuAndNE, all} は Decoder の 188 ops が NE に dispatch されている

### 1.2 HiFi-GAN

| precision | computeUnit 設定 | CPU | GPU | NE |
|---|---|---:|---:|---:|
| F32 | cpuAndGPU | 7 | 189 | 0 |
| F16 | cpuAndGPU | 8 | 188 | 0 |
| F16 | cpuAndNE | 115 | 0 | 81 |
| F16 | all | 11 | 124 | 61 |
| Int8 | cpuAndNE | 115 | 0 | 81 |
| Int8 | all | 11 | 124 | 61 |

事実:
- F16 / Int8 × cpuAndNE: HiFi-GAN の 81 ops が NE、115 ops が CPU
- F16 / Int8 × all: HiFi-GAN は CPU/GPU/NE に分散
- F32 × cpuAndGPU: NE dispatch なし

### 1.3 Encoder

| precision | computeUnit 設定 | CPU | GPU | NE |
|---|---|---:|---:|---:|
| F32 | cpuAndGPU | 0 | 101 | 0 |
| F16 | cpuAndNE | 7 | 0 | 97 |

## 2. Phase 1 結果と dispatch の対応

Phase 1 で `class=clipped` が観測された条件 = F16 / Int8 × {cpuAndNE, all}
Phase 4 で NE 行き op がある条件（Decoder と HiFi-GAN 両方）= F16 / Int8 × {cpuAndNE, all}

両者の条件が一致している。NE dispatch が clipping と相関する可能性が示唆される。
ただし NE dispatch が「直接の原因」かは op 単位の検証が必要で、現時点では未確定。

F32 × {cpuAndNE, all} は NE dispatch がなく Phase 1 でも clipping しなかった、という関係は観測されている。

## 3. HiFi-GAN の NE 行き op 内訳（F16 × cpuAndNE）

| op | cpuAndGPU (clipping なし) | cpuAndNE (clipping あり) |
|---|---|---|
| conv | 73 GPU, 1 CPU | 17 NE, 57 CPU |
| leaky_relu | 65 GPU, 4 CPU | 18 NE, 51 CPU |
| add | 44 GPU | **44 NE** |
| conv_transpose | 1 GPU, 3 CPU | 1 NE, 3 CPU |
| mul | 4 GPU | 1 NE, 3 CPU |
| tanh (最終) | 1 GPU | 1 CPU |

観測:
- `add` 44 op が cpuAndNE では全て NE に dispatch される
- `conv` は 17/74 だけが NE 行き、残り 57 は CPU
- `leaky_relu` は 18/69 だけが NE 行き、残り 51 は CPU
- 最終 `tanh` は NE 非対応なのか CPU 配置

候補（断定はしない）:
- residual connection に対応する `add` が全て NE 行きなので、residual 加算周辺で振幅差が出ている可能性
- `conv` 17 個と `leaky_relu` 18 個も NE 経路にあり、これらも候補に含まれる
- どの op / どのブロックで振幅差が始まるかは op 単位の中間出力比較がないと特定できない

## 4. Decoder と HiFi-GAN の予備切り分け

Phase 1 で各 run の `summary.txt` から `postnet_output` と `waveform_predeemph` を比較。

| 設定 | postnet max | postnet mean | predeemph rms | predeemph max |
|---|---:|---:|---:|---:|
| F16 × cpuAndGPU | 0.7314 | 0.2694 | 0.0268 | 0.36 |
| F16 × cpuAndNE | 0.7363 | 0.2689 | 0.1140 | 0.96 |
| Int8 × cpuAndGPU | 0.7290 | 0.2652 | 0.0283 | 0.52 |
| Int8 × cpuAndNE | 0.7305 | 0.2645 | 0.1171 | 0.99 |

観測:
- Decoder の postnet 出力（HiFi-GAN への入力）は cpuAndGPU と cpuAndNE で min/max/mean が小数点 2 桁レベルでほぼ一致
- HiFi-GAN 出力（predeemph）は cpuAndNE 側で rms が 4 倍程度、max が 2.7 倍程度になっている
- de-emphasis フィルタ通過後（postdeemph）の倍率も同様の傾向

解釈（現時点）:
- Decoder 出力時点では cpuAndGPU と cpuAndNE で差が小さく、Decoder 由来の可能性は低い
- HiFi-GAN 通過後に振幅差が出ているため、HiFi-GAN 内部での振幅差が生じている可能性が高い
- de-emphasis フィルタが主因とは考えにくい（cpuAndGPU と cpuAndNE で同じフィルタを通している）

## 5. PyTorch reference 比較

`scripts/generate_hifigan_reference.py` で PyTorch CPU F32 HiFi-GAN を実行（入力は iOS F16×cpuAndGPU run の `postnet_output.npy`）。

| 出力 | max | rms |
|---|---:|---:|
| PyTorch CPU F32 tanh_out | 0.3346 | 0.0269 |
| iOS F16×cpuAndGPU predeemph | 0.36 | 0.0268 |
| iOS F16×cpuAndNE predeemph | 0.96 | 0.1140 |

観測:
- PyTorch CPU F32 reference と iOS F16×cpuAndGPU は値域・rms ともに近い
- iOS F16×cpuAndNE は reference から外れる

## 6. PyTorch HiFi-GAN 内部の値域（参考情報）

`scripts/generate_hifigan_reference.py` の中間 tensor から、PyTorch CPU F32 でも HiFi-GAN 内部にはかなり大きな値が出る区間がある。

| 層 | min | max | rms |
|---|---:|---:|---:|
| upsample_3_resblock_1 | -33.67 | 11.30 | 1.51 |
| upsample_3_resblock_2 | -521.38 | 5.41 | 25.63 |
| upsample_3_stage_out | -177.43 | 4.34 | 8.75 |
| pre_post_leaky | -1.77 | 4.34 | 0.16 |
| conv_post_out | -0.50 | 0.35 | 0.027 |

これは PyTorch CPU F32 でも観測される事実。F16 でも dynamic range（±65504）内に収まる値域。ANE 経路がこの中間値域でどのように振る舞うかは op 単位の中間出力比較が必要。

## 7. 現時点の解釈

- F16 / Int8 × {cpuAndNE, all} で clipping が観測される（Phase 1）
- 同じ条件で Decoder と HiFi-GAN の両方に NE dispatch がある（Phase 4）
- 振幅差は HiFi-GAN 通過後に生じており、Decoder 由来の可能性は低い
- HiFi-GAN の `add` 44 op が全て NE 行き、`conv`/`leaky_relu` の一部も NE 行き
- どの op / どのブロックで振幅差が始まるかは未確定（op 単位の中間出力比較が必要）

## 8. 次に確認すること

- HiFi-GAN を upsample stage ごとに切った多出力 mlpackage を作り、cpuAndNE で各段の中間出力を取得して PyTorch reference と diff する
- もしくは PyTorch 上で「add の片側を 2.5x スケール」「特定 stage の出力を 2.5x スケール」などの摂動実験を行い、どこを変えると最終振幅が ANE 観測と一致するかを見る
- Phase 1 の wav と PyTorch reference を聴感比較して、振幅以外（位相、周波数特性）にも違いがあるかを確認

## 9. 注意

- 「ANE が悪い」とは断定できない。NE dispatch と clipping の相関は観測されているが、ANE 経路が直接の原因かは未確定
- F32 は ANE 非対応なので「F32 × cpuAndNE / all で動いている」のは CPU/GPU fallback の結果
- 本番アプリでの実用判断（F16 × cpuAndGPU 採用、F16/Int8 × {cpuAndNE, all} を外す等）と原因の op 特定は別の問題として扱う
