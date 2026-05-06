# 量子化 Pareto ベースライン計測（シミュレータ）

精度 (Float32 / Float16 / Int8) × 計算デバイス (cpuOnly / cpuAndGPU / cpuAndNE / all) の
12 通りで PronounSE 合成を回し、**サイズ・速さ・音質劣化** を1表にまとめた最初のスナップショット。

研究計画 [research-plan](../2026-04-22/research-plan.md) Phase 1 のベースライン取得。

## 計測環境

| 項目 | 内容 |
|---|---|
| デバイス | iOS シミュレータ (iPhone 17 Pro / iOS 26.2 / Mac M4 Max ホスト) |
| 入力 | `input_sample.wav` (バンドルされたサンプル音声) |
| 試行回数 | 各組み合わせ **1回** |
| 計測手段 | `testCaptureAllCombinations` (XCUITest) で12通りを連続実行 |

**シミュレータ値の制約（重要）**:
- ANE は実在せず CPU フォールバック → `cpuAndNE` ≈ `cpuOnly` になる
- GPU は Mac の Metal で iPhone 実機の GPU と挙動が違う
- **絶対値は研究データに使えない**。本表は計測機構の動作確認と相対比較用

## 結果

基準: **Float32 / cpuOnly** （mel L1/L2/Cos はこの mel に対する差）

| Precision | Device | totalMs | RTF | dec/step | modelMB | melL1 | melL2 | melCos |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Float32 | cpuOnly    | 12988 | 4.27 | 48.28 | 181.5 | 0.000 | 0.000 | 1.0000 |
| Float32 | cpuAndGPU  |  8101 | 2.66 | 29.94 | 181.5 | 0.000 | 0.000 | 1.0000 |
| Float32 | cpuAndNE   | 13228 | 4.35 | 49.28 | 181.5 | 0.000 | 0.000 | 1.0000 |
| Float32 | all        |  8678 | 2.85 | 32.10 | 181.5 | 0.000 | 0.000 | 1.0000 |
| Float16 | cpuOnly    |  8422 | 2.77 | 31.43 |  90.9 | 0.841 | 1.337 | 0.9997 |
| Float16 | cpuAndGPU  |  8036 | 2.64 | 29.33 |  90.9 | 0.624 | 0.945 | 0.9999 |
| Float16 | cpuAndNE   |  8339 | 2.74 | 31.19 |  90.9 | 0.841 | 1.337 | 0.9997 |
| Float16 | all        |  8096 | 2.66 | 29.82 |  90.9 | 0.624 | 0.945 | 0.9999 |
| Int8    | cpuOnly    |  8579 | 2.82 | 32.04 |  45.9 | 2.618 | 3.869 | 0.9977 |
| Int8    | cpuAndGPU  |  7679 | 2.52 | 28.23 |  45.9 | 2.944 | 4.366 | 0.9970 |
| Int8    | cpuAndNE   |  8528 | 2.80 | 31.84 |  45.9 | 2.618 | 3.869 | 0.9977 |
| Int8    | all        |  7524 | 2.47 | 27.65 |  45.9 | 2.944 | 4.366 | 0.9970 |

列の意味:
- `totalMs`: encoder + decoder全ステップ + hifigan の `predict()` 合計時間 (ms)
- `RTF`: Real-time factor = `totalMs / outputDurationMs`。1.0 未満ならリアルタイム合成可
- `dec/step`: Decoder 1 ステップあたり平均 (ms)
- `modelMB`: 使用 3 モデル (Encoder + Decoder + HiFi-GAN) の `.mlmodelc` 合計 (MB)
- `melL1` / `melL2` / `melCos`: 出力 mel の `Float32/cpuOnly` に対する L1 / L2 / コサイン類似度

## 観察

### サイズ

- **Float32: 181.5 MB → Float16: 90.9 MB → Int8: 45.9 MB**（ほぼ半分ずつ）
- 量子化はそのまま .mlmodelc サイズに反映される。Pareto の「サイズ軸」はクリーンに取れる。

### 速さ

- 全組み合わせで **RTF ≈ 2.5〜4.4** → シミュレータ上ではどれもリアルタイム合成不可。
- `Float32 + cpuAndNE` ≈ `Float32 + cpuOnly` ≈ 13秒（ANE フォールバックの典型）
- `Float32 + cpuAndGPU` ≈ `Float32 + all` ≈ 8秒（Metal GPU が支配的）
- Float16 / Int8 のあいだに有意な速度差なし（シミュレータ上では）。実機 ANE での差は別計測が必要。

### 音質劣化（mel L1）

- Float16: **0.62〜0.84**（コサイン類似度 0.9997 以上、聴感上の差は出にくいレベル）
- Int8: **2.6〜2.94**（Float16 の **約3倍**、コサイン類似度 0.997）
- Int8 でも Cos > 0.99 は保たれているので「壊れる」ほどではないが、Pareto 上で Float16 に対する明確なトレードオフが見えている。

### デバイス選択の影響パターン

`cpuOnly` と `cpuAndNE` の出力 mel が **完全一致**（L1/L2/Cos が同値）。
`cpuAndGPU` と `all` も同様に完全一致。
→ シミュレータでは ANE = CPU フォールバック、`all` ≒ GPU 経路、というのが mel 値からも裏が取れる。

## 計測機構の状況

| 取れているもの | 場所 |
|---|---|
| 出力 mel (.npy + .png) × 12 | `result/mel/` |
| 入力 mel (.npy + .png) × 1 | `result/mel/input_mel.*` |
| Timing JSON (predict時間 + RTF + modelSize) × 12 | `result/timing/` |
| 出力 wav × 12 (Precision_ComputeUnit ラベル命名) | `result/output_<P>_<U>.wav` |

集計コマンド:

```bash
PronounSE/venv/bin/python scripts/aggregate_metrics.py
```

## 制約・既知の問題

- **試行回数1回**: シミュレータの timing は数百 ms オーダーでブレる。3〜5回平均が望ましい
- **入力1サンプルだけ**: PronounSE のサンプル音声1本ぶん。複数話者・複数長さで取らないと汎化は語れない
- **音質指標が mel ドメインのみ**: 波形ドメイン (SNR / PESQ / STOI) や MCD は未取得
- **メモリ消費未計測**: Pareto に載せたいなら Instruments 連携が要る
- **シミュレータの絶対値は研究データに使えない**（前述）

## 次にやること候補

優先度順:

1. **実機計測**（最優先）— ANE 経路を含む真の速さを取る
2. **複数試行・複数音声入力** — シミュレータでも傾向の安定性を確認
3. **波形ドメインの音質指標** — Float32 を reference に SNR / MCD / PESQ
4. **モデルロード時間** — 冷起動 UX の指標
5. **メモリピーク** — Instruments の Allocations もしくは `os_proc_available_memory`

## 関連

- 計測機構の実装: [timing-measurement](../2026-04-25/timing-measurement.md)
- UI テスト周りの修正: [ui-test-loop-fixes](../2026-04-25/ui-test-loop-fixes.md)
- 集計スクリプト: [`scripts/aggregate_metrics.py`](../../scripts/aggregate_metrics.py)
- 研究計画: [research-plan](../2026-04-22/research-plan.md)
