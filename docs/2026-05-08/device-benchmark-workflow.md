# 実機 12 通り計測ワークフロー（2026-05-08〜）

iPhone 実機で `testCaptureAllCombinations` を走らせて、12 通り（Float32/Float16/Int8 × cpuOnly/cpuAndGPU/cpuAndNE/all）の timing/mel/wav を取り、ホスト側で **集計（CSV）と図（mel スペクトログラム / 振幅波形）** までを 1 コマンドで完走させる。

シミュレータ計測は研究データに使えない（ANE が無く、計測値が実機と乖離する）ため、本研究の数字はすべて実機で取る必要がある。

## 1 コマンドで完走

```bash
./scripts/run_device_benchmark.sh
```

`xcrun devicectl list devices` で `connected` 状態の iPhone を 1 台自動検出する。明示指定したい場合：

```bash
./scripts/run_device_benchmark.sh --device "iPhone (3)"
./scripts/run_device_benchmark.sh --device 44D024A4-DF7F-5FE5-8143-CAA045000A45  # CoreDevice identifier
./scripts/run_device_benchmark.sh --device 00008110-0008396C26C3401E             # ハードウェア UDID
```

`--device` には名前 / UDID / CoreDevice identifier いずれを渡しても OK（内部でデバイス名に正規化される）。

## 出力先

`run_device_benchmark.sh` を 1 回回すと、以下が同時に生成される：

| 場所 | 中身 | 永続性 |
|---|---|---|
| `result/timing/*.json` | 12 通りの計測 JSON（実機が書き出した生データ） | 次回 extract で `rsync --delete` され上書き |
| `result/mel/*.npy` `*.png` | 12 通りの出力 mel + 入力 mel | 同上 |
| `result/output_<P>_<U>.wav` | 12 通りの出力 wav | 同上 |
| `metrics/metrics_<デバイス>_<日時>.csv` | Precision × ComputeUnit の 9 列 CSV | **残る**（履歴蓄積） |
| `figures/<デバイス>_<日時>/mel_grid.png` | 入力 + 12 通り mel のグリッド画像 | **残る** |
| `figures/<デバイス>_<日時>/waveform_grid.png` | 入力 + 12 通り振幅波形のグリッド画像 | **残る** |

`metrics/` と `figures/` は `.gitignore` 対象（生計測データ・派生物のため）。レポートに貼るときは手動でコピー。

## 内部で何が起きるか

`run_device_benchmark.sh` は次の 4 ステップを順に叩く：

### 1/4. `xcodebuild test` で XCUITest を実機実行

```bash
xcodebuild test \
  -project ios/CoreMLAudioApp/CoreMLAudioApp.xcodeproj \
  -scheme CoreMLAudioApp \
  -destination "platform=iOS,name=<デバイス名>" \
  -only-testing:CoreMLAudioAppUITests/CoreMLAudioAppUITests/testCaptureAllCombinations
```

`platform=iOS` ＝ 実機（シミュレータなら `platform=iOS Simulator`）。`testCaptureAllCombinations` は既存の XCUITest で、12 通りの精度 × 計算デバイスを順番に切り替えて合成し、各回 timing JSON / mel npy / wav を `Documents/Result/` に書き出す。

### 2/4. `xcrun devicectl device copy from` で実機からファイル吸い出し

```bash
xcrun devicectl device copy from \
  --device <デバイス名> \
  --domain-type appDataContainer \
  --domain-identifier erika.com.CoreMLAudioApp \
  --source /Documents/Result \
  --destination <一時ディレクトリ>
```

Xcode 15+ で導入された `devicectl` の機能。シミュレータ用の `xcrun simctl get_app_container` の実機版に相当する。`appDataContainer` ＝ アプリのサンドボックスの `Documents/` ルート。一時ディレクトリに落としたあと、`rsync -av --delete` で `result/` に複製する。

### 3/4. `aggregate_metrics.py --csv` で集計＋ CSV 保存

```bash
PronounSE/venv/bin/python scripts/aggregate_metrics.py \
  --csv metrics/metrics_iPhone_3_20260509_000013.csv
```

`result/timing/*.json` と `result/mel/*.npy` を読んで、Precision × ComputeUnit の 12 行表を標準出力＋ CSV に書き出す。CSV は `metrics/` 配下にタイムスタンプ付きで残る。

CSV の列：`precision, computeUnit, totalMs, rtf, decAvgMs, modelMB, melL1, melL2, melCos`。

### 4/4. `view_mel.py` / `view_waveform.py` で図を生成

```bash
PronounSE/venv/bin/python scripts/view_mel.py      --save figures/<dir>/mel_grid.png
PronounSE/venv/bin/python scripts/view_waveform.py --save figures/<dir>/waveform_grid.png
```

それぞれ `result/mel/*.npy` と `result/output_*.wav` から **入力 + 12 通り** をグリッドで描画して PNG 保存する。レイアウトは 4 行 × 4 列：1 行目に入力、2〜4 行目に 3 精度 × 4 デバイス。

## 個別スクリプトの使い方

ベンチマーク全体を回さず、個別に叩きたいときの早見表：

### 集計（CSV だけ取り直す）

```bash
PronounSE/venv/bin/python scripts/aggregate_metrics.py                      # 標準出力のみ
PronounSE/venv/bin/python scripts/aggregate_metrics.py --csv path/to.csv    # CSV にも保存
```

`result/timing/` と `result/mel/` が揃っていれば動く。

### mel スペクトログラム可視化

```bash
PronounSE/venv/bin/python scripts/view_mel.py                                          # ウィンドウ表示
PronounSE/venv/bin/python scripts/view_mel.py --save figures/mel.png                   # PNG 保存
PronounSE/venv/bin/python scripts/view_mel.py output_mel_Float16_cpuAndGPU.npy         # 1 ファイルだけ
PronounSE/venv/bin/python scripts/view_mel.py output_mel_Float16_cpuAndGPU.npy --save figures/single.png
```

### 振幅波形可視化

```bash
PronounSE/venv/bin/python scripts/view_waveform.py                                     # ウィンドウ表示
PronounSE/venv/bin/python scripts/view_waveform.py --save figures/wave.png             # PNG 保存
PronounSE/venv/bin/python scripts/view_waveform.py output_Float16_cpuAndGPU.wav        # 1 ファイルだけ
```

入力波形は `ios/CoreMLAudioApp/CoreMLAudioApp/Input/input_sample.wav` を読み、グリッドの 1 行目に表示する。

### 実機からファイルだけ吸い出し（テスト再実行なし）

```bash
./scripts/extract_ui_test_results.sh --device "iPhone (3)"
```

XCUITest を回したあと、再ビルドせずに `Documents/Result/` だけホストに落としたいとき。`--device` を省くとシミュレータから取りに行く。

## 前提

- Xcode 15+（`xcrun devicectl` 利用のため）
- iPhone を USB 接続し、Mac と paired 済み（`xcrun devicectl list devices` で `connected` 表示）
- iPhone に CoreMLAudioApp をインストール済み（`xcodebuild test` が自動的に再ビルド・インストールするので 1 回目から OK）
- `PronounSE/venv` が用意済み（集計と図生成に必要）

## トラブルシューティング

### `connected` な iPhone が見つからない

```bash
xcrun devicectl list devices
```

で State 列を確認。`available (paired)` だけでは自動検出が通らないので、USB ケーブルを挿し直して `connected` にする。

### `xcodebuild` と `devicectl` で UDID の体系が違う

同じデバイスでも `xcodebuild` はハードウェア UDID（`00008110-...`）、`devicectl` は CoreDevice identifier（`44D024A4-...`）と別 ID を期待する。`run_device_benchmark.sh` の中ではどちらの入力を渡しても **デバイス名（"iPhone (3)" 等）** に正規化してから両方に渡しているので意識不要。手動で叩くときだけ注意。

### `WavFileWarning: Chunk (non-data) not understood, skipping it.`

`scipy.io.wavfile` が iOS の wav に含まれるメタデータチャンク（タイムスタンプ等）を読み飛ばしているだけの警告。データ本体は問題なく読めるので無視して OK。

### 1 回計測の数値がブレる

GPU 経路（`cpuAndGPU` / `all`）はランタイムのメモリ確保や cache の当たり方で実行ごとに数値が揺れる（mel L1 が ±数の単位で変わる）ことが観測されている。比較用の数字は **複数回計測の平均と分散** を取る運用が望ましい。

## 関連

- 計測機構の中身: [`docs/2026-04-25/timing-measurement.md`](../2026-04-25/timing-measurement.md)
- XCUITest フレーク対策: [`docs/2026-04-25/ui-test-loop-fixes.md`](../2026-04-25/ui-test-loop-fixes.md)
- 12 通りベースライン（シミュレータ）: [`docs/2026-04-27/quantization-pareto-baseline.md`](../2026-04-27/quantization-pareto-baseline.md)
- メモリ管理視点での先生コメント: [`docs/2026-05-07/hirai-comment-memory-management.md`](../2026-05-07/hirai-comment-memory-management.md)
