# 実機 12 通り計測ワークフロー（2026-05-08）

iPhone 実機で `testCaptureAllCombinations` を走らせて、12 通り（Float32/Float16/Int8 × cpuOnly/cpuAndGPU/cpuAndNE/all）の timing/mel/wav を取り、ホスト側で集計するまでを 1 コマンドで完走させる。

シミュレータ計測は研究データに使えない（ANE が無く、計測値が実機と乖離する）ため、本研究の数字はすべて実機で取る必要がある。

## 1 コマンドで完走

```bash
./scripts/run_device_benchmark.sh
```

`xcrun devicectl list devices` で `connected` 状態の iPhone を 1 台自動検出する。明示指定したい場合：

```bash
./scripts/run_device_benchmark.sh --device 44D024A4-DF7F-5FE5-8143-CAA045000A45
./scripts/run_device_benchmark.sh --device "iPhone (3)"
```

## 内部で何が起きるか

`run_device_benchmark.sh` は次の 3 ステップを順に叩いている：

### 1. `xcodebuild test` で XCUITest を実機実行

```bash
xcodebuild test \
  -project ios/CoreMLAudioApp/CoreMLAudioApp.xcodeproj \
  -scheme CoreMLAudioApp \
  -destination "platform=iOS,id=<UDID>" \
  -only-testing:CoreMLAudioAppUITests/CoreMLAudioAppUITests/testCaptureAllCombinations
```

`platform=iOS` ＝ 実機（シミュレータなら `platform=iOS Simulator`）。`testCaptureAllCombinations` は既存の XCUITest で、12 通りの精度 × 計算デバイスを順番に切り替えて合成し、各回 timing JSON / mel npy / wav を `Documents/Result/` に書き出す。

### 2. `xcrun devicectl device copy from` で実機からファイル吸い出し

```bash
xcrun devicectl device copy from \
  --device <UDID> \
  --domain-type appDataContainer \
  --domain-identifier erika.com.CoreMLAudioApp \
  --source /Documents/Result \
  --destination <一時ディレクトリ>
```

これは Xcode 15+ で導入された `devicectl` の機能。シミュレータ用の `xcrun simctl get_app_container` の実機版に相当する。アプリのサンドボックス（`appDataContainer`）の中の `Documents/Result` を丸ごとホスト側に持ってくる。一時ディレクトリに落としたあと、既存の `rsync -av --delete` で `result/` に複製する。

### 3. `aggregate_metrics.py` で集計

```bash
PronounSE/venv/bin/python scripts/aggregate_metrics.py
```

`result/timing/*.json` と `result/mel/*.npy` を読んで、Precision × ComputeUnit の 12 行表を標準出力に書き出す（速さ・サイズ・mel 差分）。

## 前提

- Xcode 15+（`xcrun devicectl` 利用のため）
- iPhone を USB 接続し、Mac と paired 済み
- iPhone に CoreMLAudioApp をインストール済み（`xcodebuild test` で自動的に再ビルド・インストールされる）
- `PronounSE/venv` が用意済み（集計用）

## トラブルシューティング

### `connected` な iPhone が見つからない

```bash
xcrun devicectl list devices
```

で State 列を確認。`available (paired)` だけだと自動検出は通らないので、USB ケーブルを挿し直して `connected` にする。

### `extract_ui_test_results.sh` 単体で吸い出しだけしたい

XCUITest はもう走らせ済みで、再ビルドせずファイルだけ取り直したいときは：

```bash
./scripts/extract_ui_test_results.sh --device "iPhone (3)"
```

`--device` を省くとシミュレータから取りに行くので注意。

### CSV で欲しい

現状 `aggregate_metrics.py` は標準出力に整形した表を `print` するだけで CSV は吐かない。レポート向けに必要になったら `--csv <path>` オプションを追加する想定。

## 関連

- 計測機構の中身: [`docs/2026-04-25/timing-measurement.md`](../2026-04-25/timing-measurement.md)
- XCUITest フレーク対策: [`docs/2026-04-25/ui-test-loop-fixes.md`](../2026-04-25/ui-test-loop-fixes.md)
- 12 通りベースライン（シミュレータ）: [`docs/2026-04-27/quantization-pareto-baseline.md`](../2026-04-27/quantization-pareto-baseline.md)
