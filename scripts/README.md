# scripts 索引

計測・変換・評価まわりのスクリプト一覧。5グループに分かれている。
計測手順と結果の本文は `docs/README.md` を目次に辿る。

実機計測の全体は `run_device_benchmark.sh` が親玉（吸い出し→集計→可視化まで自動）。

## ① 変換（PyTorch → CoreML .mlpackage）

| スクリプト | 役割 | 主な入出力 |
|---|---|---|
| `convert_transformer.py` | Encoder/Decoder を .mlpackage 化 | in: Transformer chkpt / out: `Transformer_*.mlpackage` |
| `convert_hifigan.py` | HiFi-GAN を .mlpackage 化（float16/32/int8 × shape 4モード） | in: HiFiGAN chkpt / out: `HiFiGAN_Generator_*.mlpackage` |
| `synthesis_coreml.py` | 変換後の通し合成 + PyTorch版と比較 | in: input.wav / out: `result/*_coreml_*.wav` |

`/convert` skill が `convert_transformer.py` / `convert_hifigan.py` を呼ぶ。

## ② 実機計測の制御（shell）

| スクリプト | 役割 |
|---|---|
| `run_device_benchmark.sh` | **親玉**。実機で12通り XCUITest を回し、吸い出し→集計→可視化まで全自動 |
| `extract_ui_test_results.sh` | アプリの `Documents/Result/` をホストの `result/` にコピー（シミュ/実機両対応） |
| `extract_manual_run.sh` | 手動操作の結果を `audio/` に日時付きで退避（auto-test と混ざらないように） |

`run_device_benchmark.sh` と `extract_manual_run.sh` は `extract_ui_test_results.sh` を呼ぶ。

## ③ 計測値の集計・音質評価

| スクリプト | 役割 | 出力 |
|---|---|---|
| `aggregate_metrics.py` | 12通りの 速さ/サイズ/mel差 表を生成 | stdout, `result/metrics.csv` |
| `compare_mcd.py` | baseline と各 WAV の MCD（音色差）を計算 | stdout |
| `aggregate_stability_matrix.py` | **Phase 1**：12組合せの NaN/clipping 分類 | `data/*.csv`, `docs/*.md` |
| `aggregate_compute_plan.py` | **Phase 4**：op単位の dispatch 集計 | `docs/*.md` |

`aggregate_*` の md 出力は docs に直接書き込まれる（stability-matrix-results.md, mlcomputeplan-dispatch-map.md）。

## ④ 参照値の生成・比較（Phase 3 = 内部分解デバッグ）

| スクリプト | 役割 |
|---|---|
| `generate_decoder_reference.py` | PyTorch Decoder の各層中間値を .npy 保存 |
| `generate_hifigan_reference.py` | PyTorch HiFi-GAN の中間値を .npy 保存 |
| `compare_decoder_reference.py` | 参照値 vs iOS出力 の層ごと乖離点を検出 |
| `compare_hifigan_paths.py` | PyTorch / CoreML(CPU/GPU/ANE) 経路を並列比較 |

### ステージ境界の段階切り分け（2026-06-21）

破綻が encoder / decoder / mel / HiFi-GAN / 最終wave のどこで起きるかを境界横断で特定する。
本文は [`docs/2026-06-21/ane-stage-isolation-breakdown.md`](../docs/2026-06-21/ane-stage-isolation-breakdown.md)。

| スクリプト | 役割 |
|---|---|
| `extract_pytorch_stages.py` | PyTorch baseline の5境界（mel/encoder/postnet/HiFi-GAN出力/最終wave）を .npy+.wav で保存（真値リファレンス）。iOS debug snapshot と同名ファイルなので直接比較できる |
| `compare_stages.py` | reference dir vs run dir を境界横断で比較。shape/min/max/mean/std/MAE/RMSE/cosine/peak/RMS/NaN・Inf＋4段階判定を md 出力。`--plots` で差分ヒートマップ。reference に実機良runを置けば computeUnit だけに差を帰属できる混入なしの切り分けになる |
| `compare_hifigan_isolation.py` | 同一 postnet を固定入力に PyTorch HiFi-GAN と CoreML HiFi-GAN 6条件（F32 cpuOnly / F16・Int8 × {cpuAndGPU, cpuAndNE}）を Mac 実行して比較。実機 ANE rms も併記 |
| `stage_metrics.py` | 上記2スクリプトが共有する比較メトリクス・判定ロジック（直接実行はしない） |

## ⑤ 補助・可視化

| スクリプト | 役割 |
|---|---|
| `compare_runs.py` | 複数 `audio/` アーカイブ間の md5/hash 比較 |
| `npy_to_wav.py` | debug snapshot の `waveform_postdeemph.npy` → 再生用 WAV |
| `view_mel.py` | `result/mel/*.npy` を可視化（単体 / 12通りグリッド） |
| `view_waveform.py` | `result/output_*.wav` を波形表示 |
| `dump_mel_csv.py` | mel .npy → CSV（Numbers/Excel 用） |

## compare_* の使い分け（名前が似てるので注意）

- `compare_mcd.py` … **WAV** の音色差（MCD）
- `compare_runs.py` … **アーカイブ間**の md5/hash
- `compare_decoder_reference.py` … **tensor(npy)** の層ごと乖離
- `compare_hifigan_paths.py` … **実行経路**の統計比較
- `compare_stages.py` … **5ステージ境界**を横断比較（どの段で壊れるか）
- `compare_hifigan_isolation.py` … **HiFi-GAN 単体**をエンジン別に比較（入力固定）
