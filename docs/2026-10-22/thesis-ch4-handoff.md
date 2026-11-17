# 引き継ぎメモ：卒論第4章の計測（2026-10-22 時点）

別の Claude セッションがこの作業を続けるためのメモ。結果そのものは
[`thesis-ch4-benchmark-report.md`](thesis-ch4-benchmark-report.md) を先に読むこと。

## 1. 何をしていたか

- 卒論（`~/repo/g2353702/BachelorThesis/setupPaper/paper_temp/Chapters/Chapter4/index.md`）第4章 4-2 節に載せるデータを取るため、
  iOS アプリ（`ios/CoreMLAudioApp`）に UI を使わない計測モードを実装し、実機で 12 条件 × 4 入力を計測した。
- 元の指示書（手順 1〜6）: `/private/tmp/claude-501/-Users-sakuraierika-repo-g2353702-BachelorThesis/09b97e6c-9ef4-4c8a-8869-a1bc12f3df12/scratchpad/thesis-ch4-benchmark-prompt.md`
  （tmp なので消えている可能性がある。要点は本メモと報告にある）
- **手順 1〜6 はすべて完了**。PR: https://github.com/orukahairuka/coreml-audio-research/pull/26
  （ブランチ `feature/thesis-ch4-benchmark`、main は PR #25 の 0 初期化修正を含む）

## 2. いまの状態と、残っていること

| 項目 | 状態 |
|---|---|
| 実装・単体テスト | 完了。実機（iPhone 13）で 42 件成功 |
| iPhone 13 の本番計測 | 完了（全成功）。`metrics/thesis_ch4_iPhone_3_20261021_175344/` |
| iPhone 17 Pro の本番計測 | 完了。`metrics/thesis_ch4_iPhone_17_Pro_20261021_225625/`（vocoder の 1 起動を合意のうえやり直し済み） |
| 表・図・README | 両端末ぶん生成済み（各ディレクトリの `tables/`、`figures/`、`README.md`） |
| 報告・4-1 節で直す点の一覧 | `docs/2026-10-22/thesis-ch4-benchmark-report.md` |
| PR #26 | 作成済み。レビュー・マージはまだ |

残っている可能性があること（ユーザーから頼まれたら着手する。勝手に始めない）:

- PR #26 のレビュー対応（bot の指摘への返信は `reply-bot-review` スキルがある）
- 4-2 節の本文に合わせた表・図の調整（列の選び方、図の体裁）
- 卒論本文の修正は**ユーザーが行う**。こちらからは書き換えない（指示書の注意事項）

## 3. ファイルの場所

| 種類 | 場所 |
|---|---|
| 計測モード（アプリ） | `ios/CoreMLAudioApp/CoreMLAudioApp/Models/Thesis/`（Config・Recorder・Pipeline・BenchRunner）、`Views/ThesisBenchView.swift`、`App/CoreMLAudioAppApp.swift` |
| 入力（gitignore） | `ios/CoreMLAudioApp/CoreMLAudioApp/Input/thesis_src_{1,6,11,2}.wav`、`thesis_pytorch_postnet_*.npy`、`thesis_inputs.json`。`scripts/make_thesis_inputs.py` で再生成できる |
| PyTorch 基準（gitignore） | `data/thesis_ch4/pytorch_reference/<入力>/` |
| PC スクリプト | `scripts/make_thesis_inputs.py` → `scripts/run_thesis_bench.py` → `scripts/thesis_ch4_metrics.py` → `scripts/aggregate_thesis_ch4.py` |
| 計測データ（gitignore） | `metrics/thesis_ch4_<端末>_<日時>/`（`raw/` に生データ、`raw/device/` が端末から吸い出したもの） |
| 失敗した試行（残してある） | `metrics/thesis_ch4_iPhone_17_Pro_20261021_213916_failed_untrusted/`、`..._220114_aborted_thermal/`、リハーサル `metrics/thesis_ch4_rehearsal_iPhone_3_20261021_174024/` |
| iPhone 17 Pro のアプリの Documents のバックアップ | `audio/iPhone_17_Pro_documents_backup_20261021_220000/`（入れ直す前に吸い出したもの） |

## 4. 計測の回し方

```bash
cd ~/repo/g2353702/coreml-audio-research
PronounSE/venv/bin/python scripts/make_thesis_inputs.py          # 入力と PyTorch 基準（初回だけ）

# Release ビルド（-allowProvisioningUpdates が必要）
cd ios/CoreMLAudioApp && xcodebuild build -allowProvisioningUpdates -project CoreMLAudioApp.xcodeproj \
  -scheme CoreMLAudioApp -configuration Release -destination 'platform=iOS,name=iPhone (3)' -derivedDataPath <dd>
cd ../..

# load / plan / infer（条件の前に thermal を待つ）
caffeinate -i python3 -u scripts/run_thesis_bench.py --device "iPhone (3)" --install <dd>/Build/Products/Release-iphoneos/CoreMLAudioApp.app \
  --phases load,plan,infer --session <ID> --out-dir metrics/thesis_ch4_<端末>_<日時>/raw
# vocoder（thermal を待たない。同じセッションの続きなので --launch-start で起動 ID をずらす）
caffeinate -i python3 -u scripts/run_thesis_bench.py --device "iPhone (3)" --phases vocoder --no-thermal-wait \
  --session <同じID> --launch-start 1001 --out-dir <同じ raw/>

PronounSE/venv/bin/python scripts/thesis_ch4_metrics.py metrics/thesis_ch4_<端末>_<日時>/raw
PronounSE/venv/bin/python scripts/aggregate_thesis_ch4.py metrics/thesis_ch4_<端末>_<日時>
```

- 端末名（devicectl）: iPhone 13 =「iPhone (3)」、iPhone 17 Pro =「櫻井絵理香のiPhone」
- iPhone 17 Pro は nominal に戻らないので `--thermal-ok-states nominal,fair` を付けた
- 1 条件だけ流すときは `--conditions Float32:cpuOnly --inputs 1`。条件の順は cpuOnly → cpuAndGPU → cpuAndNE → all
- 所要時間: iPhone 13 で約 3.5 時間（うち thermal の待ちが約 2 時間）、iPhone 17 Pro で約 1.7 時間

## 5. ハマりどころ（再発しやすいもの）

- **iOS 27 シミュレータでは単体テストが動かない。** XCTest のホストアプリがデバッガ接続待ち（`T` 状態）で止まり、
  `unable to connect to "com.apple.instruments.deviceservice.lockdown"` が出る（macOS 26 + iOS 27 beta ランタイム）。
  単体テストは実機で `xcodebuild test -allowProvisioningUpdates ... -destination 'platform=iOS,name=iPhone (3)' -parallel-testing-enabled NO` で流す。
  シミュレータでの動作確認は `run_thesis_bench.py --simulator <UDID>`（simctl で直接起動）で行える
- **実機向けビルドには `-allowProvisioningUpdates` が必要**（無いとプロファイルが見つからないエラー）
- **iPhone 17 Pro で「CoreMLAudioApp はもう利用できません」**: 開発元（Apple Development: irukahanemurusikanai@gmail.com、無料チーム・プロファイルは 7 日で切れる）が未信頼。
  アプリを削除して入れ直すと「設定 → 一般 → VPN とデバイス管理」に項目が出るので、ユーザーに「信頼」を押してもらう
- **起動し直しただけでは「初回の読み込み」にならない**（Core ML がディスクに準備結果を残す）。インストールし直すと初回になる。
  インストールし直しても `Documents/` は残る
- `devicectl device install` はまれに即座に失敗するので、スクリプトでやり直している
- **`.gitignore` の `audio/`** は macOS（core.ignorecase=true）では `Models/Audio/` にも一致していた。`/audio/` に直してある
- `pkill -f` のパターンに `iPhone (3)` を入れると `()` が正規表現として扱われて一致しない。PID で止める
- PronounSE の venv には pandas がない。スクリプトは numpy と標準ライブラリだけで書いてある

## 6. ユーザーとの約束ごと（このセッションで決まったこと）

- 区切りごとに止まって報告し、OK をもらってから次に進む
- 計測中に異常終了・時間切れ・NaN が出ても、**勝手に条件を外したりやり直したりしない**。記録して報告し、やり直すかは聞く
- 計画から変えるときは理由を説明して合意を取る。合意済みの変更: load を 3 種類に分ける、vocoder は thermal を待たない、
  iPhone 17 Pro は fair 以下で始める、vocoder の失敗した 1 起動だけやり直す
- コミットは日本語・プレフィックスなし・修正ごとに分ける。ブランチは `feature/`。PR 前に動作確認（`.claude/rules/git-workflow.md`）
- 返答は日本語
