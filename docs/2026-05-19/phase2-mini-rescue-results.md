# Phase 2 mini — 救済実験 観測（2026-05-19）

ブランチ: `feature/fp32-gpu-memory-timing`
デバイス: iPhone 13（iOS 26.5）
日付: 2026-05-19〜20（実機計測は 2026-05-19 23:51 開始、~11 分）
関連: [`coreml-hifigan-investigation-handoff.md`](coreml-hifigan-investigation-handoff.md), [`stability-matrix-analysis.md`](stability-matrix-analysis.md)

Phase 1 で観測した F16/Int8 × {cpuAndNE, all} の clipping に対し、2 系統の救済策の効果を計測した記録。Phase 2 の本来計画は warm-up / retry / fallback 系（戦略 A〜I）の効果率測定だが、今回は (J) 出力後処理 と (K) HiFi-GAN だけ別エンジンに逃がす の 2 つに絞った。

## 1. 試した戦略

| ID | 戦略 | 対象 | 期待 |
|---|---|---|---|
| J | 出力波形を後処理で再正規化（peakNorm / rmsNorm / fixedGain ×0.25） | F16/Int8 × cpuAndNE | 振幅レベルだけは `normal_loud` 域に落とせるか |
| K | Decoder は cpuAndNE のまま、HiFi-GAN だけ cpuAndGPU で再ロード | F16/Int8 | HiFi-GAN を NE 経路から外せば clipping が消えるか |

実装: `Phase2RescueTests.strategyJ_*` / `strategyK_*`。Strategy K は `AudioSynthesizer.reloadHifigan(precision:computeUnits:shapeMode:)` を追加して実現した。

Phase 1 でも観測済みの「F16/Int8 × cpuAndNE Repeat3 が iter1/2/3 すべて bit-identical」（CSV 上の `waveform_postdeemph_sha` 一致）はそのまま新規実験せずに引き継いだ。同一インスタンス内で warm-up や retry を入れても改善しにくいことを意味する観測としてだけ参照する。

## 2. 観測値

### 2.1 Strategy J（出力正規化）

`*_raw` は AudioSynthesizer の出力をそのまま測ったもの（Phase 1 と一致）。`*_peakNorm` / `*_rmsNorm` / `*_fixedGain025` は `outputWaveform` を test 側で再正規化した結果。

| 設定 | tag | int16_rms | int16_peak | wav clip+/− 数 | classification |
|---|---|---:|---:|---:|---|
| F16 × cpuAndNE | raw           | 13289.5 | 71468.9 | 1090 / 821 | clipped |
| F16 × cpuAndNE | peakNorm (0.95)| 5788.3 | 31128.7 | 0 / 0 | normal_loud |
| F16 × cpuAndNE | rmsNorm (baseline 5029) | 5029.0 | 27045.3 | 0 / 0 | normal_loud |
| F16 × cpuAndNE | fixedGain ×0.25 | 3322.4 | 17867.2 | 0 / 0 | normal_loud |
| Int8 × cpuAndNE | raw          | 14154.9 | 87400.0 | 1075 / 1099 | clipped |
| Int8 × cpuAndNE | peakNorm (0.95)| 5041.4 | 31128.7 | 0 / 0 | normal_loud |
| Int8 × cpuAndNE | rmsNorm (baseline 5029) | 5029.0 | 31051.8 | 0 / 0 | normal_loud |
| Int8 × cpuAndNE | fixedGain ×0.25 | 3538.7 | 21850.0 | 0 / 0 | normal_loud |

- `clip+/−` 数は `npy_to_wav.py` で float → int16 化したときに ±1.0 を超えてクランプされたサンプル数。raw は 2.8〜3.2% のサンプルが ±1.0 をはみ出している。
- 3 通りの正規化はいずれも `normal_loud` 域に収まる。レベル計測上は救えている。
- ただし数値だけでは audio quality は判定できない。clip した区間に対応する float 値が「真の波形 × 大きすぎるスケール」なのか「op の飽和で歪んだ値」なのかは音を聴かないと分からない。
- wav は `audio/iPhone_3_phase2mini_20260519/playable/*phase2_J_*.wav` に書き出してある。`scripts/npy_to_wav.py` 経由で生成。

### 2.2 Strategy K（HiFi-GAN だけ cpuAndGPU で再ロード）

`AudioSynthesizer.loadModels(precision, cpuAndNE, fixed262)` → `reloadHifigan(precision, cpuAndGPU, fixed262)` の順で組合せ、`synthesize()` を 3 回。

| 設定 | tag | int16_rms | int16_peak | clip+/− | classification |
|---|---|---:|---:|---:|---|
| F16 (dec=NE, hifi=GPU) | iter1 | 4993.7 | 25974.3 | 0 / 0 | normal_loud |
| F16 (dec=NE, hifi=GPU) | iter2 | 4993.7 | 25974.3 | 0 / 0 | normal_loud |
| F16 (dec=NE, hifi=GPU) | iter3 | 4993.7 | 25974.3 | 0 / 0 | normal_loud |
| Int8 (dec=NE, hifi=GPU) | iter1 | 5431.0 | 27447.9 | 0 / 0 | normal_loud |
| Int8 (dec=NE, hifi=GPU) | iter2 | 5431.0 | 27447.9 | 0 / 0 | normal_loud |
| Int8 (dec=NE, hifi=GPU) | iter3 | 5431.0 | 27447.9 | 0 / 0 | normal_loud |

3 iter ともすべて bit-identical（int16 値が完全一致）。HiFi-GAN の cpuAndGPU 経路自体は Phase 1 で確定論的だったので、整合的な結果。

### 2.3 Decoder 出力が cpuAndNE 単独ケースと bit-identical なことの確認

K で「Decoder は確かに cpuAndNE で回っているか」を sha256 で照合した。

| 比較 | postnet_output sha（先頭 10 文字） | 一致 |
|---|---|---|
| Phase 1 F16 × cpuAndNE 単独 | `dc8df52b33` | — |
| Phase 2 K F16 (dec=NE, hifi=GPU) | `dc8df52b33` | ✓ 一致 |
| Phase 1 Int8 × cpuAndNE 単独 | `0b61fb6512` | — |
| Phase 2 K Int8 (dec=NE, hifi=GPU) | `0b61fb6512` | ✓ 一致 |
| 参考: Phase 1 F16 × cpuAndGPU 単独 | `a8fabfd8d1` | （K と不一致、Decoder dispatch が違うので当然） |
| 参考: Phase 1 Int8 × cpuAndGPU 単独 | `33c43a8790` | （K と不一致） |

→ K の `reloadHifigan` セットアップは意図どおり Decoder=cpuAndNE のまま HiFi-GAN だけ別エンジンに載せられている。

## 3. 現時点の解釈（推論、断定はしない）

### 3.1 「HiFi-GAN の NE dispatch と clipping の相関」がより支持される

- 同じ Decoder 出力（postnet sha = `dc8df52b33` / `0b61fb6512`）に対し、HiFi-GAN を cpuAndNE で動かすと clipped、cpuAndGPU で動かすと normal_loud になった。
- Decoder 出力は K と Phase 1 cpuAndNE 単独で bit-identical。違うのは HiFi-GAN の dispatch だけ。
- ここから「F16/Int8 × cpuAndNE で出る clipping は HiFi-GAN の NE 経路と相関している」とは言える。
- ただし「ANE が悪い」「特定 op が原因」とまでは言えない。op 単位の中間出力比較がまだ無い。

### 3.2 出力後処理（J）について

- 振幅レベルだけ見れば `normal_loud` 域に落とせる。
- ただし `npy_to_wav` で raw を変換した際に 2.8〜3.2% のサンプルが ±1.0 を超えていたことから、raw の段階で「音として正しいまま振幅だけ大きい」のか「op の飽和で歪んだ波形」なのかが分からない。
- 音質判定には wav を試聴する必要がある。`audio/iPhone_3_phase2mini_20260519/playable/` 配下の wav で比較する。

### 3.3 Phase 5 への含意（仮）

- F16/Int8 × cpuAndNE / all を本番から外すことの妥当性はさらに支持された。
- ANE が必須の本番要件は今のところ無いので、K（HiFi-GAN だけ GPU 退避）を採用する強い動機もまだ無い。
- もし「Decoder は ANE が高速で動いてほしい」という別の要件が出てきたら、K のセットアップは選択肢になる。

これらの方針案は Phase 5 で再レビューする。

## 4. 残課題

- **J の音質聴感判定**: `audio/iPhone_3_phase2mini_20260519/playable/` の wav を試聴し、peakNorm / rmsNorm / fixedGain いずれが「歪みなく聞こえる」か（あるいは全部歪んでいるか）を確認する。歪みなく聞こえれば「raw の clipping は単なる線形スケール拡大」だと結論できる。歪んでいれば、HiFi-GAN NE 内部で何らかの非線形な飽和が起きている可能性が示唆される。
- **K の音質聴感判定**: F16/Int8 × (dec=NE, hifi=GPU) の wav と Phase 1 F16/Int8 × cpuAndGPU 単独の wav を比較。同等であれば「HiFi-GAN を GPU に逃がすだけで完全に救える」と言える。
- **op 単位の原因特定**: 今回は op レベルの中間出力を取らない。重い作業なので Phase 3 で扱う場合に決める。

## 5. 関連ファイル

- 実装: `ios/CoreMLAudioApp/CoreMLAudioApp/Models/Synthesis/AudioSynthesizer.swift`（`reloadHifigan` 追加）
- テスト: `ios/CoreMLAudioApp/CoreMLAudioAppTests/Phase2RescueTests.swift`（戦略 J / K のテスト 4 個追加）
- 計測ログ: `_tmp_logs/phase2_mini_20260519_235054.log`（gitignore 対象）
- debug snapshot: `audio/iPhone_3_phase2mini_20260519/debug/`（14 run）
- 再生用 wav: `audio/iPhone_3_phase2mini_20260519/playable/`（14 wav）
