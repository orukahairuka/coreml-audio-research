# 通常操作の応答時間の計測（合成の開始から再生開始 API の呼び出しまで）

卒論第4章の End-to-End 時間は計測用の処理（`ThesisPipeline`）で測った合成処理の時間であり、
画面の「合成」操作（`SynthesisViewModel.runSynthesis()`）の応答時間ではない。その差を埋めるための計測。

- 状態: **両機種とも計測済み**（iPhone 17 Pro は 2 セッション、iPhone 13 は 1 セッション）
- 論文本文への反映: **実施済み**（4-1節(5)、4-5節(5)・表4-16、4-6節(3)〜(5)、第1・3・5章、要旨、おわりに）

## 1. 通常操作と計測用の処理の違い

| 処理 | 計測用の処理（ThesisPipeline） | 通常操作（AudioSynthesizer / SynthesisViewModel） |
|---|---|---|
| 入力の読み込み | 1 回 | 3 回（波形用・メル用・表示用） |
| メル抽出・Encoder・Decoder・HiFi-GAN・デエンファシス | 共通（`AudioFeatureExtractor` / 各 Runner） | 共通 |
| Decoder の進捗通知 | MainActor に移るが中身は空 | MainActor で `status` と `progress` を更新（画面が描き直される） |
| 各段の出力の統計（ArrayStats・RMS・ログ 1 行） | なし | あり |
| 表示用のメル（入力・出力） | なし | あり（出力は 262×256 個を 1 つずつ取り出す） |
| メル・時間記録のファイル保存 | なし | あり（`saveMelArtifacts` / `saveTimingArtifact`） |
| WAV の保存と再生 | なし | `AudioPlayer.play()` の中で AVAudioSession の設定 → WAV 保存 → `AVAudioPlayer` の生成・準備 → `play()` |

Decoder の調査用の統計（環境変数で有効化）と入力のまとめて複写は `DecoderRunner` で共通なので、最適化 1・2 は通常操作にも効いている。

## 2. 計測点

時計は `DispatchTime.now().uptimeNanoseconds`（単調増加）。

| 点 | 位置 |
|---|---|
| entry | `runSynthesis()` の先頭 |
| A | `loadModels` から戻った直後（読み込み済みなので確認だけ）＝合成処理の開始 |
| B | `synthesizer.synthesize()` から戻った直後 |
| C | `saveMelArtifacts`・`saveTimingArtifact` の完了 |
| D | `audioPlayer.play()` を呼ぶ直前 |
| E | `AudioPlayer.play()` 内、WAV の保存が終わった直後 |
| F | `AVAudioPlayer.play()`（再生開始の API）を呼ぶ直前 |
| G | `AVAudioPlayer.play()` から戻った直後 |
| end | 再生終了の通知を MainActor で受けた時刻 |

依頼時の想定（C＝合成音の保存完了、D＝play 呼び出し）と実際の順序が違う。**合成音の WAV は `play()` の中で保存される**ので、
WAV の保存は D→E に入る。「再生開始 API の呼び出しまで」は **A→F**。スピーカーから音が出始める時刻ではない。

## 3. 実装

- `Models/ResponseBench/ResponseTimeProbe.swift`、`ResponseBenchConfig.swift`、`ViewModels/ResponseBenchDriver.swift`（新規）
- `SynthesisViewModel.swift`（計測点 entry・A〜D・end、`loadModelsForCurrentSelection()`）、`AudioPlayer.swift`（E〜G）、`ContentView.swift`（`.task` で起動）
- `scripts/run_response_bench.py`（起動・thermal 待ち・吸い出し）、`scripts/aggregate_response_bench.py`（集計）

`CMLA_RESPONSE_BENCH=1` のときだけ、通常の画面のまま `runSynthesis()` を入力ごとに ウォームアップ 1 回 → 本計測 3 回 呼ぶ。
入力は同梱の `thesis_src_<name>.wav` を `.recording(URL)` として選ぶ（入力1〜4 の順、再生終了から次の合成まで 1000 ms 待つ）。
モデルは計測の前に読み込む。CSV と SHA-256 は再生終了の後（計測区間の外）で書く。計測モードでないときは計測点は何もしない。

## 4. 実行手順

```bash
# Release ビルド
cd ios/CoreMLAudioApp
xcodebuild -project CoreMLAudioApp.xcodeproj -scheme CoreMLAudioApp -configuration Release \
  -destination 'generic/platform=iOS' -derivedDataPath <dd> -allowProvisioningUpdates build
cd ../..
# iPhone 13（thermal は nominal まで待つ。卒論第4章と同じ）
scripts/run_response_bench.py --device "iPhone (3)" --install <dd>/Build/Products/Release-iphoneos/CoreMLAudioApp.app
# iPhone 17 Pro（計測していなくても fair から下がらないことがあるので fair まで許す）
scripts/run_response_bench.py --device <UDID> --thermal-ok-states nominal,fair
# 集計
scripts/aggregate_response_bench.py metrics/response_<端末>_<日時> \
  --pipeline metrics/thesis_ch4v4_decinput_<端末>_20261118_001315/bulk_noprofile/raw/device/infer_runs.csv
```

- 合成音は端末のスピーカーから鳴る。計測中は端末に触らない（下の 1 回目のように値が乱れる）
- iPhone 13 は Wi-Fi 経由（Xcode の Devices で Connect via network）か USB でつなぐ。Bluetooth では不可

## 5. 結果（iPhone 17 Pro、iOS 27.0.0、Release、Float16・条件 C）

| セッション | 状態 |
|---|---|
| `metrics/response_iPhone_17_Pro_20261118_144104` | 16 回すべて成功・thermal nominal。**計測中に端末を閉じた**。入力1 の本計測 3 回目だけ A→B 2,045 ms・推論 1,736 ms（他の回の約 2 倍） |
| `metrics/response_iPhone_17_Pro_20261118_144331` | 端末に触らずに取り直したもの。16 回すべて成功・thermal nominal。**こちらを本番とする** |

2 回目の本計測 3 回の中央値（ms）:

| 入力 | 合成音 | A→B | B→C | D→E | E→F | F→G | **A→F** | 推論（モデル呼び出しの合計） | パイプラインの E2E（最終・区間記録なし） |
|---|---|---|---|---|---|---|---|---|---|
| 入力1 | 3.04 s | 1,083 | 6.2 | 1.7 | 7.2 | 79.9 | **1,098** | 865 | 945 |
| 入力2 | 3.04 s | 1,084 | 6.5 | 1.7 | 7.2 | 79.9 | **1,099** | 861 | 946 |
| 入力3 | 2.15 s | 836 | 5.1 | 1.7 | 7.2 | 80.0 | **850** | 660 | 711 |
| 入力4 | 1.59 s | 673 | 4.1 | 1.6 | 7.1 | 80.0 | **686** | 523 | 543 |

- 出力波形の SHA-256 は 2 セッションとも、4 入力すべてで計測用の処理（ch4-v4・iPhone 17 Pro）と一致した → 計測点を入れても合成結果は変わらず、通常操作とパイプラインは同じ波形を出す
- entry→A（読み込み済みの確認）は 0.2 ms 未満、C→D は 0.01 ms 未満
- A→B はパイプラインの E2E より 130〜140 ms 長い。**推論時間自体もパイプラインの記録（全入力の中央値 727 ms）より長い**。
  通常操作では Decoder の各回で画面の進捗を更新するなど処理が違うが、原因は調べていない（同じ日の別セッションの比較なので、セッション差の可能性も残る）
- `AVAudioPlayer.play()` の呼び出しから戻るまで（F→G）が毎回約 80 ms
- A→F は 4 入力すべてで合成音の長さより短い

## 5b. 結果（iPhone 13、iOS 27.2.0、Release、Float16・条件 C）

`metrics/response_iPhone_13_20261118_163452`: 16 回すべて成功・thermal は計測中ずっと nominal。端末には触っていない。

本計測 3 回の中央値（ms）:

| 入力 | 合成音 | A→B | B→C | D→E | E→F | F→G | **A→F** | 推論（モデル呼び出しの合計） | パイプラインの推論 | パイプラインの E2E |
|---|---|---|---|---|---|---|---|---|---|---|
| 入力1 | 3.04 s | 2,262 | 13.3 | 2.6 | 6.9 | 53.8 | **2,284** | 1,765 | 1,724 | 1,992 |
| 入力2 | 3.04 s | 2,373 | 12.2 | 2.3 | 7.2 | 57.8 | **2,395** | 1,884 | 1,778 | 2,059 |
| 入力3 | 2.15 s | 1,785 | 8.2 | 2.2 | 7.1 | 58.3 | **1,803** | 1,405 | 1,321 | 1,520 |
| 入力4 | 1.59 s | 1,349 | 7.1 | 2.4 | 7.0 | 59.1 | **1,365** | 1,039 | 986 | 1,133 |

- 出力波形の SHA-256 は 4 入力すべてで計測用の処理（ch4-v4・iPhone 13）と一致した
- 入力3 の本計測 2 回目だけ A→B 2,421 ms・推論 2,010 ms（他の 2 回より約 640 ms 長い）。thermal は nominal のまま。原因は不明。中央値には影響しない
- A→F は 4 入力すべてで合成音の長さより短い（入力4: 1,365 ms／1.59 s が最も近い）
- A→B はパイプラインの E2E より 216〜314 ms 長い。うち推論時間の差が 41〜106 ms、残りは推論以外（入力の読み込み 3 回・表示用の計算・統計・進捗の更新など。内訳は測っていない）
- F→G は約 54〜59 ms（iPhone 17 Pro は約 80 ms）
- 同じ OS（27.2.0）でのパイプライン計測と比べられる。iPhone 17 Pro はパイプライン計測と同じ iOS 27.0.0

## 6. 論文に反映するときの下書き

- 4-1 節(5): 「通常操作の応答時間」を定義に加える。区間 A〜G の位置、WAV 保存が `play()` 内にあること、A→F は再生開始 API の呼び出しまでで音が出る時刻ではないこと、計測手順（条件 C・最終の実装・Release・入力ごとにウォームアップ 1 回＋本計測 3 回・モデル読み込み済み・入力1〜4 の順・再生終了後 1 秒待つ・thermal の扱い）
- 4-5 節に項を追加: 表「通常操作の応答時間」（入力ごとの A→B、保存（B→C・D→E）、E→F、F→G、A→F、パイプラインの E2E、合成音の長さ）。
  iPhone 17 Pro: 5 節の表。iPhone 13: 5b 節の表
- 4-6 節(2)「利用者の待ち時間」: 「計測していない」を実測値に置き換える。ただし A→F は音が出る時刻ではないこと、通常操作の推論時間がパイプラインと異なったことはそのまま書く
- 4-6 節(3)・(4)、第5章の制約、要旨: 「通常操作の時間は未計測」を外し、残る制約（音が出るまでの時間は測っていない、1 セッション・条件 C のみ）に置き換える
- iPhone 13 で A→F が合成音の長さを超える入力があれば、そのまま書く
