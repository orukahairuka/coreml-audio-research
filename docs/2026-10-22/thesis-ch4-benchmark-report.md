# 卒論第4章の計測：実装と実機計測の報告

卒論第4章「評価と考察」の 4-2 節に載せるデータを取るため、iOS アプリに UI を使わない計測モードを実装し、
iPhone 13（主な計測）と iPhone 17 Pro（確認用）で 12 条件を計測した。ブランチは `feature/thesis-ch4-benchmark`。

計測データ（表・図・生データ・README）は `metrics/` に置いてある（gitignore 対象）。

| 端末 | ディレクトリ | iOS | 成否 |
|---|---|---|---|
| iPhone 13（`iPhone14,5`） | `metrics/thesis_ch4_iPhone_3_20261021_175344/` | 27.2.0 | 全起動・全実行が成功 |
| iPhone 17 Pro（`iPhone18,1`） | `metrics/thesis_ch4_iPhone_17_Pro_20261021_225625/` | 27.0.0 | vocoder の 1 起動が通信の切断で失敗し、合意のうえでその 1 起動だけやり直した |

## 1. 実装したもの

| 種類 | ファイル | 内容 |
|---|---|---|
| アプリ | `Models/Thesis/ThesisBenchConfig.swift`、`ThesisRecorder.swift`、`ThesisPipeline.swift`、`ThesisBenchRunner.swift`、`Views/ThesisBenchView.swift`、`App/CoreMLAudioAppApp.swift` | 起動時の環境変数 `CMLA_THESIS_BENCH=load|plan|infer|vocoder|thermal` で、UI を使わずに 1 つのモードを実行して `exit` する。1 回の実行を 1 行として CSV に追記し、npy・wav・crumb を残す |
| アプリ | `Models/Audio/WavWriter.swift`（`AudioPlayer` から切り出し） | 再生せずに同じ経路で Int16 の WAV を保存する |
| アプリ | `WaveformStatistics`、`DecoderRunner` / `EncoderRunner` / `VocoderRunner` | \|x\|>1 の列、Decoder のステップごとの prediction 時間、入力配列の NaN/Inf 個数 |
| PC | `scripts/make_thesis_inputs.py` | 入力 4 本のダウンロードと、PyTorch 版 Decoder の出力メル（262 フレームに揃えた形）・HiFi-GAN の出力（基準）の生成 |
| PC | `scripts/run_thesis_bench.py` | devicectl でアプリを起動し直しながら計測し、起動ごとの成否・thermal の待ち時間を記録する |
| PC | `scripts/thesis_ch4_metrics.py`、`scripts/aggregate_thesis_ch4.py` | RMS比・Peak比・コサイン類似度・WAV 飽和率を計算し、4-2 節の表（中央値 [最小〜最大] (n)）と図を作る |

## 2. 計測の手順（実際に行ったもの）

1. フェーズは load → plan → infer → vocoder。各フェーズで cpuOnly → cpuAndGPU → cpuAndNE → all の順（ANE を使う条件を後）。
2. **load**: 条件ごとに「アプリを入れ直してから起動」× 3、「起動し直すだけ」× 2。各起動で最初の読み込み 1 回と、解放してからの再読み込み 5 回。
3. **plan**: 条件ごとに 1 起動で 3 モデルの MLComputePlan を保存。
4. **infer**: 条件ごとに 1 プロセス。入力 1 → 6 → 11 → 2 の順に、ウォームアップ 1 回（読み込み直後の 1 回目は「初回の推論」）＋定常時 5 回。
5. **vocoder**: 条件ごとに 1 プロセス。HiFi-GAN だけを読み込み、PyTorch 版 Decoder の出力メル（入力を先頭 T′ フレームに切ってから Decoder を回し、262 フレームに 0 埋め）を入れる。回数は infer と同じ。
6. **thermal**: 条件を変える前に、iPhone 13 は nominal、iPhone 17 Pro は fair 以下まで下がるのを待った。vocoder はどちらも待たなかった。
7. アプリは Release ビルド。

## 3. わかったこと

数値は iPhone 13 の定常時の中央値。iPhone 17 Pro も傾向は同じ（波形の指標はほぼ同じ値）。

1. **振幅の異常は F16/Int8 × cpuAndNE・all の 4 条件だけで起き、両端末で同じ値になる。**
   - HiFi-GAN 出力の RMS 比（対 Float32 × cpuOnly）は 1.5〜4.4、コサイン類似度はほぼ 0。デエンファシス後の \|x\|>1 の割合は 1.2〜26%、最大値は 2.0〜2.8。
   - Decoder の出力メルのコサイン類似度は 0.99 以上で、HiFi-GAN の段で生じている。
   - PyTorch のメルを入れた HiFi-GAN だけの比較でも、同じ 4 条件だけ RMS 比が 2.2〜4.0（17 Pro は 2.3〜4.1）。ほかの 8 条件は 0.99〜1.01。
   - MLComputePlan では、この 4 条件の HiFi-GAN の一部の op が Neural Engine に割り当てられる計画だった（cpuAndNE: 81 op、all: 61 op）。Float32 × cpuAndNE は全 op が CPU の計画で、出力は Float32 × cpuOnly とビット単位で一致した。
2. **範囲の超過はほかの 8 条件でも起きる。** 入力 11 と 2 では Float32 × cpuOnly でも \|x\|>1 が 0.3〜0.8%。PyTorch 版でも同じように超えるので、モデルそのものの性質。
3. **WAV の保存はクリップ（ラップアラウンドではない）。** 1.5・3.0・100 は 32767、負の値は −32768 になる（単体テスト `WavWriterTests`）。±1.0 ちょうども 32767/−32768。WAV の飽和率は \|x\|>1 の割合とほぼ一致した。
4. **RTF が 1 を下回るのは、iPhone 13 では F16/Int8 × cpuAndNE・all の 4 条件（0.54〜0.61）だけ。** F16/Int8 × cpuOnly は 0.97、Float32 × cpuOnly は 2.3、cpuAndGPU と Float32 × all は 3.5〜3.7。iPhone 17 Pro では F16/Int8 の 8 条件がすべて 1 を下回る（0.30〜0.65）。
5. **読み込み時間は、インストール直後の初回だけが長い。** iPhone 13 では、ANE を使わない条件で 0.45〜2.6 秒、F16/Int8 × cpuAndNE・all で 6.6〜10.8 秒（大半が HiFi-GAN）。起動し直しただけの初回は 6 ms〜670 ms、再読み込みは 3.6〜150 ms。
6. **End-to-End 時間の約 2/3 はモデルの呼び出し以外。** iPhone 13・入力 1 で E2E は 10.7〜20.4 秒、そのうち 9〜10 秒は入力配列の準備などモデルの呼び出し以外の時間（Decoder の各ステップで最大 262×256 個の値を `NSNumber` で詰め直す処理など）。前処理は約 30 ms、デエンファシスは 1 ms 未満。
7. **同じ条件・同じ入力の出力は毎回同じ。** 6 回の出力波形の SHA-256 はすべて一致した。

## 4. 予想と違ったこと

- **起動し直すだけでは初回の読み込みにならない。** Core ML が端末向けの準備結果をディスクに残すため、プロセスを起動し直しただけの初回は数 ms で終わる。リハーサルで判明し、インストールし直す起動を加えた。
- **iOS の T が PyTorch より全入力で 4 フレーム少ない。** iOS: 262/292/185/137、PyTorch: 266/296/189/141。入力 6 の T′ は 262 で変わらないが、入力 11 と 2 の T′ は 185 と 137。
- **iPhone 13 では GPU を使う条件が CPU だけより遅い**（RTF 3.5〜3.7 と 1〜2.3）。
- **ANE を使う条件のインストール直後の読み込みが、2026-06-10 の計測（iOS 27.0、35〜68 秒）より短い**（6.6〜10.8 秒、iOS 27.2.0）。クラッシュもなかった。
- **Float32 × cpuAndGPU の 1 回目だけ結果が変わる現象は再現しなかった**（全実行の SHA-256 が一致）。
- **熱の影響が大きい。** iPhone 13 では条件の前に nominal まで待っても、infer の実行の前の状態は serious 146 回・fair 97 回・nominal 45 回（288 回中）だった。待ち時間は合計約 2 時間。リハーサル（冷えた状態）の Float32 × cpuOnly の RTF は 1.55、本番は 2.28。
- **iPhone 17 Pro は計測していなくても 30 分間 fair のままで nominal に戻らなかった。** fair 以下で始めることにした。

## 5. 本文の 4-1 節で直す必要がある点（本文は書き換えていない）

| 4-1 節 | 本文の記述 | 実際 |
|---|---|---|
| (2) 表 4-1 | iPhone 13・iPhone 17 Pro とも iOS 27.0 | iPhone 13 は **iOS 27.2.0**、iPhone 17 Pro は **iOS 27.0.0** |
| (2) 表 4-1 | — | アプリは Release ビルド。計測は devicectl で起動し直しながら行った |
| (3) | 筆者が録音した口真似を 5 本程度、前処理後 262 フレーム以上、配列はすべて書き込まれる | ASJ152 デモの口真似 4 本（1, 6, 11, 2）。iOS の前処理後の T は 262/292/185/137 で、**入力 11 と 2 は T′ < 262**（残りは 0 で埋める） |
| (3) | 入力に用いるメルは未確定 | PyTorch 版で、入力を先頭 T′ フレームに切ってから Decoder を回した出力メルを 262 フレームに 0 埋めしたもの。基準は同じメルを入れた PyTorch 版 HiFi-GAN（CPU, float32）の出力 |
| (3) | 0 初期化の修正後に計測（予定） | 修正（PR #25）後に計測した。3-5 節の記述も合わせて直す必要がある |
| (4) 読み込み | 「初回の読み込み」「再読み込み」の 2 種類、各条件 3 回程度 | **3 種類**: インストール直後の初回（3 回）、起動し直しただけの初回（2 回）、再読み込み（25 回）。準備結果の再利用についての注記（要出典）を、数値（数 ms）で裏づけられる |
| (4) 推論 | 定常時は 3〜5 回、ウォームアップ 1 回で安定するかは未確認 | 定常時 5 回。初回の推論とウォームアップ後の差は表 4-2-2 を参照 |
| (4) E2E | 計測するか未確定 | 計測した。定義は「入力ファイルの読み込みと前処理の開始から、デエンファシスを適用した波形が得られるまで」（表示用のメルの計算と保存は含めない） |
| (5) 要確認 | AVFoundation の変換で範囲を超えた値がどう扱われるか | **クリップされる**（ラップアラウンドしない） |
| (5) 実装予定 | ±1 超過率の定義の修正など | すべて実装済み（\|x\|>1 の列、RMS比・Peak比・コサイン類似度、WAV 飽和率、端末・iOS・熱状態、初回と定常時の区別） |
| (5) 記録する情報 | 端末・iOS・入力・T・条件・回数・入力メルのハッシュ・熱状態 | 加えて出力波形の SHA-256、実行の前後の thermalState、入力配列の NaN/Inf の個数を記録した |
| (5) 熱状態 | 連続計測で温度が上がると処理が遅くなる可能性 | 条件の前に nominal（iPhone 13）／fair 以下（iPhone 17 Pro）まで待ったが、条件の中で serious まで上がった。vocoder は待たずに続けて回した |

## 6. 再現

```bash
PronounSE/venv/bin/python scripts/make_thesis_inputs.py
# Release でビルド（-allowProvisioningUpdates）して .app を作る
python3 scripts/run_thesis_bench.py --device "iPhone (3)" --install <CoreMLAudioApp.app> --phases load,plan,infer
python3 scripts/run_thesis_bench.py --device "iPhone (3)" --phases vocoder --no-thermal-wait \
    --session <同じセッション> --launch-start 1001 --out-dir <同じ raw/>
PronounSE/venv/bin/python scripts/thesis_ch4_metrics.py <計測ディレクトリ>/raw
PronounSE/venv/bin/python scripts/aggregate_thesis_ch4.py <計測ディレクトリ>
```

iOS 27 のシミュレータでは XCTest のホストアプリがデバッガの接続待ちで止まる（macOS 26 と iOS 27 beta の組み合わせ）ため、単体テストは実機（iPhone 13）で流した（42 件すべて成功）。
