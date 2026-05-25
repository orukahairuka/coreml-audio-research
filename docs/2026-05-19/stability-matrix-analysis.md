# Phase 1 安定性マトリクス 観測結果（2026-05-19 実機 iPhone 13）

生データ: [`stability-matrix-results.md`](stability-matrix-results.md), `data/2026-05-19/stability_matrix.csv`
取得元: `audio/iPhone_3_phase1_20260519/debug/`（5/17 の過去 run + 5/19 の Phase 1 run 計 78 run）

XCTest ログ: `/tmp/phase1_test.log`（5/19 のセッション分のみ）

## 1. 集約マトリクス

5/19 セッションの Repeat3 結果を 12 組合せでまとめると（同一 `AudioSynthesizer` インスタンスを使い回した iter1 / iter2 / iter3 はすべて bit-identical）:

| precision \ computeUnit | cpuOnly | cpuAndGPU | cpuAndNE | all |
|---|---|---|---|---|
| **Float32** | normal_loud (5029 / 24326) | normal_loud (5029 / 24326) <br>※ 単発初回は確率的 (clipped 8441/33162 を観測) | normal_loud (5029 / 24326) | normal_loud (5029 / 24326) |
| **Float16** | normal_loud (4911 / 25432) | normal_loud (4993 / 24176) | clipped (13289 / 71468) | clipped (13284 / 68336) |
| **Int8** | normal_loud (5607 / 28170) | normal_loud (5534 / 27710) | clipped (14154 / 87400) | clipped (14162 / 83202) |

値は (int16_rms / int16_peak)。`normal_loud` の参考値は manual baseline `(5029, 24326)`。

## 2. 観測事実

### 2.1 F16 / Int8 × {cpuAndNE, all} で clipping が観測された

決定論的（iter1/2/3 bit-identical）に発生する clipping:
- F16 × cpuAndNE: peak 71468 / rms 13289
- F16 × all: peak 68336 / rms 13284
- Int8 × cpuAndNE: peak 87400 / rms 14154
- Int8 × all: peak 83202 / rms 14162

iter1/2/3 が完全に同じ値で出るので、確率的な数値オーバーフローではなく、特定 dispatch 経路に対する決定論的な挙動と考えられる。

### 2.2 F32 × cpuAndGPU の単発初回は確率的

`directRun1`（fresh `AudioSynthesizer` 単発呼び出し）で clipped (rms 8441, peak 33162) を観測。同一インスタンスを使い回した Repeat3 はすべて normal_loud (5029, 24326) で bit-identical。

過去メモ `f32-cpuandgpu-mlmodel-2-manual-loud-2026-05-17` の観測と整合する。

### 2.3 F32 × {cpuOnly, cpuAndGPU(warm), cpuAndNE, all} は manual baseline と一致

F32 では precision × computeUnits の組合せで Repeat3 の値がすべて `(5029, 24326)` で manual baseline と一致した。

ANE は F32 を受け付けないため、F32 × cpuAndNE / all の実行は実質的には CPU / GPU で行われている（Phase 4 の MLComputePlan 観測で NE dispatch が 0 であることを確認、`compute-plan-analysis.md` 参照）。

### 2.4 F16 / Int8 は GPU 経路（cpuAndGPU）では normal_loud

- F16 × cpuAndGPU: 4993 / 24176（normal_loud）
- Int8 × cpuAndGPU: 5534 / 27710（normal_loud）

### 2.5 warm-up は F32 × cpuAndGPU で有効

`fp32GpuDummyWarmupThenReal` テスト: dummy synthesize 1 回後の本番 synthesize は rms 5029, peak 24326 で manual baseline と一致。Phase 2 救済戦略 A（dummy warm-up 1 回）の効果が観測された。

## 3. 観測と整合する解釈

- F16 / Int8 で computeUnits に NE を含めた場合に clipping が観測される
- F32 では NE 指定でも実際の dispatch は NE に行かない（Phase 4 で確認）ので、F32 × {cpuAndNE, all} は通常通り動く
- ANE 経路と clipping の対応が観測されたが、ANE 経路内のどの段階で振幅差が出ているかは Phase 4 単独では判定できない（HiFi-GAN 出力時点で振幅差が観測されていることは Phase 4 分析で別途扱う）

## 4. 暫定的な実用判断

| 設定 | 評価 | 備考 |
|---|---|---|
| F16 × cpuAndGPU | 採用候補 | 既存本番設定。manual baseline と近い |
| F32 × cpuOnly | 採用候補 | manual baseline と一致、CPU のみ |
| F32 × cpuAndGPU | 条件付き採用候補 | warm-up 1 回入れた上で採用可 |
| Int8 × cpuOnly | 採用候補 | manual baseline と近い |
| F16 × cpuAndNE / all | UI から外す候補 | clipping を観測 |
| Int8 × cpuAndNE / all | UI から外す候補 | clipping を観測 |

これは Phase 5（実装方針）で改めて確定する。

## 5. 計測条件

- デバイス: iPhone 13（iOS 26.5）
- ホスト: macOS, Xcode 26.x
- ブランチ: `feature/fp32-gpu-memory-timing`
- HiFi-GAN: `Float32/Float16_fixed262`, `Int8_HiFiGAN_Generator_int8`（legacy range16-1000）
- Transformer: `Float32/Float16_fixed262`, `Int8` legacy
- 入力: `input_sample.wav`
- ShapeMode: 各 precision で `fixed262` 使用可なら `fixed262`、なければ `range1`（Int8 のみ）
- フィルタ: `applyDeemphasis` 適用後の波形を集計

## 6. 注意

- 5/17 の過去 run（freshFirst* 系）も同じ audio archive 配下にあり、78 run 集計には含まれる。同 timestamp で見ると 5/19 セッションの結果は本ファイル §1 表の通り
- Int8 は legacy `HiFiGAN_Generator_int8`（RangeDim 16-1000）を使用しており、Int8 の HiFi-GAN は他と shape mode が異なる点に注意
- 各組合せ N=1 fresh launch のみ。launch 間の決定性を確認するには複数 launch が必要（次のラウンドで取る）

## 7. 次に確認すること

- Phase 2 救済実験（warm-up / retry / fallback）の効果計測
- Phase 4 の HiFi-GAN dispatch（[`compute-plan-analysis.md`](compute-plan-analysis.md) 参照）と組み合わせて、HiFi-GAN 内部での振幅差の発生段階を絞る
- 本ファイルの観測値を Phase 5 実装方針の入力にする
