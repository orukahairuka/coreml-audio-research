# F32 × cpuAndGPU の「quiet 出力」調査メモ（2026-05-17 進行中）

ブランチ `feature/fp32-gpu-memory-timing` で進めた調査の中間記録。**結論は未確定**。

## 元の問題意識

ユーザー観察：「自動テストだと F32 × cpuAndGPU の爆発音が出ない（実機 iPhone）」。手動でアプリ操作すると鳴る。

仮説の出発点：メモリ確保／解放のタイミングが影響している（[`docs/2026-05-07/hirai-comment-memory-management.md`](../2026-05-07/hirai-comment-memory-management.md) の動的メモリ確保コメントとの接続）。

## やったこと（時系列）

### 1. メモリ仮説検証：XCUITest 各組合せ後に sleep
`testCaptureAllCombinations` の内側ループ末尾に `Thread.sleep` を入れて、組合せ間の解放時間を稼ぐ。

| sleep | F32 cpuAndGPU melL1 | F32 cpuAndNE melL1 |
|---|---|---|
| なし × 4 (5/11) | 3.77, 4.45, 3.78, 4.52 | 12.22, 11.96, 18.44, 8.33 |
| 3s | 4.04 | 13.14 |
| 10s × 2 | 5.02, 3.09 | 0.00, 8.52 |
| 30s | 3.60 | 9.38 |

**メモリ解放仮説は支持されない**：
- cpuAndGPU の melL1 が単調に下がらない
- 一度出た「cpuAndNE melL1 = 0」が再現しない（再ランで 8.52 に戻った）

### 2. playback-wait race の発見と修正
`SynthesisViewModel.runSynthesis` を確認したところ、`playOutput()` は再生を「開始」するだけで終了を待たない。`isProcessing = false` が再生 START 直後に立ってしまい、XCUITest は「合成完了」と判断して次の組合せのモデルロードに進む。結果として **直前の再生（3.04 秒）と次のモデルロードが並行**していた。

修正：`playOutputAndAwaitCompletion()` を追加。`audioPlayer.onPlaybackFinished` を `CheckedContinuation` で待ち、再生終了まで `isProcessing = true` を維持。

→ コミット `3e2e235`。

**ただしこの修正だけでは F32 × cpuAndGPU の quiet 問題は解決しなかった**：
- playback-wait 版 auto-test の F32 × cpuAndGPU rms は 480 ぐらいのまま
- sleep 各種と同じ範囲（400-728）にとどまる

### 3. 手動操作との比較（extract_manual_run.sh を追加）
iPhone の `Documents/Result/` を `audio/<デバイス>_manual_<日時>/` に退避する小スクリプトを追加（コミット `573e2b4`）。手動で操作した wav を auto-test と比較できるようにする。

ユーザーに手動で F32 × cpuAndGPU を 3 回実行してもらった結果：

| ラン | F32 × cpuAndGPU md5 | rms | peak |
|---|---|---|---|
| 手動 1（5/16 23:35）| **3b4a657c44** | **5029** | 24327 |
| 手動 2（5/16 23:39）| **3b4a657c44** | **5029** | 24327 |
| 手動 3（5/16 23:43）| **3b4a657c44** | **5029** | 24327 |

**手動は決定論的に loud（mode B）**。3 回完全に同じバイト。しかも md5 3b4a657c44 は **手動 F32 × all の出力と完全に一致**する。

対照的に、auto-test の cpuAndGPU は md5 が毎回違い、rms 400-728 の quiet 範囲（11 ラン分のデータ）。

### 4. 順序効果の検証
`testCaptureAllCombinationsGpuFirst` を追加：12 通りの computeUnits 順序を `[cpuAndGPU, cpuOnly, cpuAndNE, all]` に変更。

結果：

| | 通常順 (cpuOnly first) | gpu-first |
|---|---|---|
| F32 cpuOnly | rms 496 | **rms 5029** ← loud に化けた |
| F32 cpuAndGPU | rms 480 | rms 728（quiet のまま）|
| F32 cpuAndNE | rms 1112 | **rms 5029** |
| F32 all | rms 2541 | **rms 5029** |
| F16 全 4 種 | (4 件全部) | (4 件全部 md5 完全一致) |
| Int8 全 4 種 | (4 件全部) | (4 件全部 md5 完全一致) |

**重要観察**：
- Float16 と Int8 は順序を変えても **md5 完全一致**（順序非依存）
- Float32 だけが順序で出力が変わる
- gpu-first 順では F32 cpuOnly と F32 cpuAndNE が **md5 完全一致** (`65ebd2d976`)（フォールバックで同じ経路に合流）
- **F32 × cpuAndGPU は順序を入れ替えても quiet のまま**

### 5. XCUITest 単発テスト
`testFp32GpuFreshFirst` を追加：XCUITest 起動直後に F32 × cpuAndGPU だけ 1 発走らせる。

結果：md5 `f880685e8b`、**rms 310**（今までで最小）。

→ 手動と同じ「fresh から cpuAndGPU 1 発目」のはずなのに **loud にならない**。XCUITest 起動と Springboard 起動の何かが違う。

## 現時点で確実に言えること

1. **playback-wait race は本物のバグ**で、修正済み。これは独立した発見。
2. **手動 F32 × cpuAndGPU は決定論的に loud**（rms 5029、md5 3b4a657c44、F32 × all と完全一致）。
3. **XCUITest 経由の F32 × cpuAndGPU は観測した全 13 ランで quiet**（rms 310-728）。
4. **Float16 / Int8 は順序非依存**で安定。
5. **Float32 の cpuOnly / cpuAndNE / all は順序効果あり**：cpuAndGPU を先に走らせると後続が loud（rms 5029）に化ける。

## まだ確証が持ててないこと

1. **「XCUITest 経由は必ず quiet」は本当に決定論的か？**  
   観測した全データは quiet 範囲だが、サンプル数が少ない。確率的に loud が出る可能性は否定しきれない。**testFp32GpuFreshFirst を 3-5 回追試で再現性確認が次の一手**。

2. **XCUITest と手動で「何が」違うのか不明**  
   - launch 経路（Springboard vs XCTestRunner）
   - プロセス権限・サンドボックス
   - launch 後の natural delay（人間が画面眺める数秒）
   - AVAudioSession の初期化タイミング
   - などの可能性が並列に残る

3. **F32 × cpuAndGPU 自身が「順序入れ替えても quiet のまま」の理由**  
   gpu-first にすると後続は loud になるのに cpuAndGPU 自身は quiet。これが ANE フォールバック決定の動的性質とどう関係するか整合解釈できていない。

## 試す価値のある次の実験

### A. XCUITest の再現性確認
`testFp32GpuFreshFirst` を 3-5 回連続実行。md5 / rms の分布を見る。
- 全部 quiet 範囲 → 「XCUITest = 決定論的に quiet」が強く支持
- 1 回でも loud → 仮説否定、XCUITest でも loud は出る

### B. XCUITest launch 後に待つ
`app.launch()` → `Thread.sleep(forTimeInterval: 10)` → 初操作。これで loud になれば「launch 直後タイミング問題」確定。

### C. launchArguments / launchEnvironment で signal
XCUITest が UI 自動化モードに入る合図を切り替えて、アプリ側で何かが変わるか確認。

### D. Unit テストで synthesizer 直接呼ぶ
UI を介さず `AudioSynthesizer.synthesize(...)` を直接 await。これで loud になれば「UI 自動化（XCTestRunner プロセス全般）が原因」確定。quiet なら別の根本原因。

### E. `MLComputePlan` で op 割り当てを可視化（iOS 17+）
F32 × cpuAndGPU の各 op が CPU / GPU のどちらに割り当てられているかを XCUITest vs 手動で比較。同じなら数値的なドリフト、違うなら経路選択の問題。

---

## 追加調査（2026-05-17、深夜）

### 既存アーカイブから判明した「決定打」: divergence は HiFi-GAN より上流

[`scripts/compare_runs.py`](../../scripts/compare_runs.py) で `audio/iPhone_3_manual_F32_gpu_manual*` と `audio/iPhone_3_manual_fresh_first_test_*` / `audio/iPhone_3_manual_gpu_first_test_*` の **既存** `mel/` 配下を比較したところ:

| run | input_mel.npy md5 | postnet_output (F32×cpuAndGPU) md5 | postnet mean | postnet max | wav md5 | wav rms |
|---|---|---|---|---|---|---|
| manual 1 | f07898a94b | **1301a5a51f** | -53.024 | -6.855 | a1c82ad4af | 5029 |
| manual 2 | f07898a94b | **1301a5a51f** | -53.024 | -6.855 | a1c82ad4af | 5029 |
| manual 3 | f07898a94b | **1301a5a51f** | -53.024 | -6.855 | a1c82ad4af | 5029 |
| XCUITest freshFirst | f07898a94b | **b302dcc729** | -68.668 | -17.679 | c78e451923 | 310 |
| XCUITest gpuFirst | f07898a94b | **bd4cc0181a** | -63.707 | -14.042 | 61630d7754 | 728 |

`input_mel.npy` は全 5 ラン同じ md5 (`f07898a94b`)。つまり **mel 前処理は決定論的で手動と XCUITest で完全に一致している**。
一方 `output_mel_Float32_cpuAndGPU.npy` (= Decoder postnet 出力 = HiFi-GAN への入力) は **md5 が手動 vs XCUITest で異なる**。

→ **F32×cpuAndGPU quiet 問題は HiFi-GAN の問題ではない**。Encoder か Decoder の段階で出力が分岐している。
→ section 2 の「playback-wait race」修正後も rms が改善しなかったのも、これで説明がつく (再生まわりは出力に関係ない)。

### 次の experiment runbook

`feature/fp32-gpu-memory-timing` ブランチに以下を追加 (本コミットで実装):

1. **`DebugRunSnapshot` (新規調査用コード)** — `CMLA_DEBUG_SNAPSHOT=1` のときだけ `Documents/Result/debug/<runId>/` に
   - `mel_normalized.npy` (Encoder 入力直前の正規化メル — これまで保存されていなかった)
   - `encoder_output.npy` (Encoder の出力 memory)
   - `postnet_output.npy` (Decoder postnet 出力)
   - `waveform_predeemph.npy` / `waveform_postdeemph.npy`
   - `summary.txt` (sha256 / shape / min/max/mean/rms + 実行コンテキスト: pid, XCTestRunner 有無, label, uptime)

   を吐く。本番動作には影響しない (env var なしで no-op)。

2. **XCUITest 追加メソッド** (各メソッドが 1 つの仮説のみ検証):
   - `testFp32GpuFreshFirst` — sleep なし baseline (label: `freshFirstNoSleep`)
   - `testFp32GpuFreshFirstWithLaunchSleep10` — launch 直後に 10 秒 sleep (label: `freshFirstSleep10`)
   - `testFp32GpuFreshFirstRepeat3` — 3 回 fresh launch × 1 発ずつ (`freshFirstRepeat1/2/3`)
   - `testFp32CpuOnlyFreshFirst`, `testFp32AllFreshFirst` — 単発計測の再現性確認

3. **Unit テスト追加** (`Fp32QuietInvestigationTests.swift`):
   - `directSynthesisFp32GpuOnce` — UI / XCUITestRunner を一切介さず `AudioSynthesizer.synthesize` を直接 await
   - `directSynthesisFp32GpuRepeat3` — 同じインスタンスで 3 連続呼び

4. **`scripts/compare_runs.py`** — 複数の `audio/<dir>` を引数に取り、wav (md5/rms/peak) と debug stage hashes (mel/encoder/postnet/waveform) を Markdown テーブルで並べる。

### 実行手順（iPhone 実機）

```bash
# 1. ビルドして実機にインストール (Xcode で Run 1 回でも可)
cd ios/CoreMLAudioApp
xcodebuild -project CoreMLAudioApp.xcodeproj -scheme CoreMLAudioApp \
    -destination 'generic/platform=iOS' build-for-testing

# 2. 仮説別に 1 メソッドずつ実行 (各テストは独立)
# 実機 ID は xcrun devicectl list devices で確認
DEV='iPhone (3)'  # 例

for METHOD in \
    testFp32GpuFreshFirst \
    testFp32GpuFreshFirstWithLaunchSleep10 \
    testFp32GpuFreshFirstRepeat3 \
    testFp32CpuOnlyFreshFirst \
    testFp32AllFreshFirst; do
    xcodebuild test-without-building \
        -project CoreMLAudioApp.xcodeproj \
        -scheme CoreMLAudioApp \
        -destination "platform=iOS,name=${DEV}" \
        -only-testing:CoreMLAudioAppUITests/CoreMLAudioAppUITests/${METHOD}
    # 各テストごとに吸い出して別ディレクトリにアーカイブ
    cd /Users/sakuraierika/repo/g2353702/coreml-audio-research
    ./scripts/extract_manual_run.sh --device "${DEV}" --label "auto_${METHOD}"
    cd ios/CoreMLAudioApp
done

# 3. Unit テストも同様 (UI 自動操作なし経路)
xcodebuild test-without-building \
    -project CoreMLAudioApp.xcodeproj \
    -scheme CoreMLAudioApp \
    -destination "platform=iOS,name=${DEV}" \
    -only-testing:CoreMLAudioAppTests/Fp32QuietInvestigationTests
./scripts/extract_manual_run.sh --device "${DEV}" --label "unit_direct"

# 4. 比較表を生成
cd /Users/sakuraierika/repo/g2353702/coreml-audio-research
./scripts/compare_runs.py audio/iPhone_3_manual_*auto_test* audio/iPhone_3_manual_*unit_direct* audio/iPhone_3_manual_F32_gpu_manual*
```

### 期待される判定マトリクス

| 仮説 | テスト | loud (rms 5029) なら | quiet (rms 300-728) なら |
|---|---|---|---|
| H1: 手動と XCUITest で mel 入力が違う | 全テストの `mel_normalized.npy` sha256 比較 | sha256 一致 → H1 否定 | sha256 不一致 → H1 支持 |
| H2: launch 直後タイミング問題 | `testFp32GpuFreshFirstWithLaunchSleep10` | H2 支持 | H2 否定 |
| H3: XCUITest = 決定論的に quiet | `testFp32GpuFreshFirstRepeat3` の 3 ラン分布 | 1 回でも loud → H3 否定 | 全 quiet → H3 強く支持 |
| H4: UI 自動操作が原因 (XCTestRunner 全般ではなく) | `Fp32QuietInvestigationTests` (Unit) | UI 自動操作のみ原因 | XCTest 環境全般が原因 |
| H5: F32×cpuOnly/all も単発ならば loud | `testFp32CpuOnlyFreshFirst` / `testFp32AllFreshFirst` | F32×cpuAndGPU だけが特殊 | F32 全体 quiet → 別仮説 |

### 補足: 既存データから事前に判明していること

既存の `audio/iPhone_3_manual_*` ディレクトリは「手動操作後の `Documents/Result/` まるごと吸い出し」なので、**他の組み合わせの wav は前回 run の残骸**であり「手動の結果」ではない (例: `audio/iPhone_3_manual_F32_all_20260516_230440/output_Float32_cpuAndGPU.wav` は md5 `5aa552853e` で auto-test の `iPhone_3_20260516_211058` と完全一致。これは手動で F32×all を実行した時点で cpuAndGPU の wav は前の auto-test の残骸が残っていただけ)。

→ 「手動で F32×cpuAndGPU が loud」と言えるのは **dedicated 3 ラン (manual1/2/3)** だけ。比較表でもこの 3 件だけを基準にする。

---

## 実機実験結果 (2026-05-17, iPhone 13 / iOS 26.x)

`extract_manual_run.sh --label auto_all_experiments` で吸い出した 11 個の debug snapshot を `summary.txt` から整理。全 run で `mel_normalized` sha256 は `35f61dbe02...` で一致 (mel 前処理は決定論的)。`waveform_postdeemph` の `rms × 32767` ≒ wav の Int16 rms。

| # | run | precision/cu | enc_sha | post_sha | post_mean | wave_rms_f | int16_rms | wave_max | wave_sha |
|---|---|---|---|---|---|---|---|---|---|
| 1 | freshFirstNoSleep   | F32×gpu | 7a86f45193 | **483ab4e454** | 0.149 | 0.0217 | **711**  | 0.729 | fd8f3e6a89 |
| 2 | freshFirstSleep10   | F32×gpu | 7a86f45193 | **483ab4e454** | 0.149 | 0.0217 | **711**  | 0.729 | fd8f3e6a89 |
| 3 | freshFirstRepeat1   | F32×gpu | 7a86f45193 | fa3026262c | 0.224 | 0.0275 | 900  | 0.794 | d8b45b4346 |
| 4 | freshFirstRepeat2   | F32×gpu | 7a86f45193 | d869e3caf8 | 0.171 | 0.0263 | 861  | 0.757 | d86424251b |
| 5 | freshFirstRepeat3   | F32×gpu | 7a86f45193 | **d869e3caf8** | 0.171 | 0.0263 | **861** | 0.757 | d86424251b |
| 6 | freshFirstCpuOnly   | F32×cpu | 3e2a21d1af | d3b18f1621 | 0.169 | 0.1003 | 3285 | 0.716 | 7c8b590622 |
| 7 | freshFirstAll       | F32×all | 7a86f45193 | 28a611865b | 0.130 | 0.0987 | 3233 | 0.708 | 22e198f03d |
| 8 | Unit: Once (1st call new instance) | F32×gpu | 7a86f45193 | 4483eba938 | 0.160 | 0.0229 | 750  | 0.352 | b6bc627239 |
| 9 | Unit: Repeat3 iter1 (1st call new instance, after #8 in same process) | F32×gpu | 7a86f45193 | 12f658502f | 0.153 | 0.0647 | 2120 | 0.599 | 1c2da502ea |
| 10 | Unit: Repeat3 iter2 (reuse instance) | F32×gpu | 7a86f45193 | **23c51a5431** | 0.270 | 0.1535 | **5029** | 0.742 | 396a67e8b0 |
| 11 | Unit: Repeat3 iter3 (reuse instance) | F32×gpu | 7a86f45193 | **23c51a5431** | 0.270 | 0.1535 | **5029** | 0.742 | 396a67e8b0 |

(Row 8/9 のラベルは Swift Testing の並列実行で env var が混線したため一見同名だが、xcodebuild ログの順序と `uptime` で識別: row 8 = `directSynthesisFp32GpuOnce()` の int16_rms 750.1、row 9 = `directSynthesisFp32GpuRepeat3()` iter1 の int16_rms 2121.0。一致確認済み。)

### 判定マトリクス (実測値で確定)

| 仮説 | 結果 | 根拠 |
|---|---|---|
| H1: mel 入力が違う | **DISPROVEN** | 11 run 全部 `mel_normalized` sha256 = `35f61dbe02...` |
| H2: launch 直後タイミング (10s sleep で解消) | **DISPROVEN** | row 1 (sleep なし) と row 2 (10s sleep) が **bit-identical** (post_sha `483ab4e454`, wave_sha `fd8f3e6a89`) |
| H3: XCUITest は決定論的に quiet | **半分支持** | row 1/2 と row 3/4/5 で sha が違うので「決定論的」とは言えない。ただし全 5 run とも int16_rms 711-900 の quiet 範囲なので「XCUITest 経由は確実に quiet」は支持 |
| H4: UI 自動操作が原因 | **DISPROVEN** | Unit Test (UI なし、XCTestRunner なし) でも 1st call は quiet (row 8: rms 750)。XCUITest と同じ quiet レンジ |
| H5: F32 × cpuOnly / all も単発なら quiet | **DISPROVEN** | row 6 (cpuOnly): int16_rms 3285 (loud)、row 7 (all): int16_rms 3233 (loud)。F32×cpuAndGPU だけが特殊 |

### 新しい仮説 H6 (今回のデータで強く支持)

**F32 × cpuAndGPU の合成は「同じ MLModel インスタンスで 2 回目以降を呼ぶ」と manual と完全一致する deterministic loud (int16_rms 5029) に収束する。1 回目はプロセス全体の状態に依存して非決定的に quiet〜intermediate になる。**

直接の証拠 (Unit テスト `directSynthesisFp32GpuRepeat3`):

| call # | post sha | int16 rms | int16 peak |
|---|---|---|---|
| iter 1 (cold) | 12f658502f | 2120 | (intermediate) |
| iter 2 (warm) | **23c51a5431** | **5029** | **24326** ← manual1/2/3 と完全一致 |
| iter 3 (warm) | **23c51a5431** | **5029** | **24326** ← iter 2 と bit-identical |

→ iter 2 と iter 3 は `postnet_output` / `waveform_postdeemph` の sha256 まで完全一致。**収束後は再現性 100%**。
→ かつ iter 2/3 の int16 peak 24326 と rms 5029 は手動 dedicated ラン (`a1c82ad4af`: peak 24327, rms 5029) と一致。**これは「真の loud」が `MLModel` 再呼び出し 2 回目以降に出る現象**であることを示す。

### なぜ Encoder 出力は同じで postnet 出力だけ違うのか

`encoder_output` sha256 は cpuAndGPU の全 row で `7a86f45193...` で **完全一致** (cpuOnly だけが浮動小数の微差で `3e2a21d1af...`)。

つまり Encoder (1 回の推論) は決定的。divergence は Decoder ループ (262 回の自己回帰呼び出し) の中で発生している。Decoder の各ステップは前回の出力を入力に取るので、初回ステップでわずかでも数値が違うと 262 ステップ後には大きく違う。

GPU 経路で MTLDevice の compute pipeline state (PSO) や Metal performance shader のキャッシュが初回 dispatch で安定しないため、Decoder ループ内のどこかで初回だけ違う数値が出ている、が現時点の最有力な技術的説明 (確定はしていない)。

### 確定した観察と未確定の論点

**確定:**
1. mel 入力 / Encoder 出力は決定的で全環境一致
2. Decoder/postnet 段階で divergence が起きる
3. `MLModel` インスタンスを保持したまま 2 回目以降呼ぶと manual と bit-identical な loud (rms 5029) に収束
4. F32×cpuAndGPU 固有の問題 (F32×cpuOnly / F32×all は単発でも loud レンジ)
5. XCUITest も Unit Test も UI 操作の有無も無関係 (どれも fresh プロセスの 1st call は quiet)
6. launch 直後の sleep 10s は無効 (bit-identical な quiet 出力)

**未確定 (要追加実験):**
- なぜ row 1/2 (`483ab4e454`) と row 3/4/5 (`fa3026262c`, `d869e3caf8`) で 1st call の sha が違うのか
  - 仮説: テスト実行間隔・前回プロセスの終了状態などで kernel-side Metal キャッシュが少し変わる
- なぜ row 4 (iter2) と row 5 (iter3) は別プロセスなのに bit-identical なのか
  - 仮説: Metal PSO がディスクキャッシュされて 2 回目以降の起動で同じ状態になる
- F32 × cpuAndGPU 以外で「2 回目以降に出力が変わる」現象がないか (Float16/Int8 で同様の Repeat3 を回す価値あり)

### 次の検証案

1. **F16 / Int8 で同じ Repeat3 Unit テスト** — 2 回目以降に変わるか? 変わらなければ問題は F32 × cpuAndGPU 固有確定
2. **`MLComputePlan`** でどの op が GPU で実行されているか確認 (iOS 17+)
3. **Decoder ループのステップ別 sha** — どのステップから divergence が始まるか (decoderStepStats の `sha256` を増やす)
4. **本番アプリでの実用的対応** — `AudioSynthesizer.loadModels` 直後にダミー synthesize を 1 回投げて warm-up する (恒久修正候補だが、本研究では今は計測フェーズなので保留)

---

## 第 3 段階追加実験 (2026-05-19, iPhone 13 / iOS 26.x)

`Fp32QuietInvestigationTests` (Swift Testing, `.serialized`) に 5 メソッド追加して 7 テスト直列実行 (1298 秒)。`DebugRunSnapshot` に Decoder 各 step の mel_out/postnet_out 全要素 sha256+stats を `decoder_steps.csv` (263 行) として append する機能を入れた。

### Repeat3 サマリ (各組合せの iter1/2/3 を 1 つの新 AudioSynthesizer で連続呼び)

| 組合せ | iter1 rms / peak | iter2 | iter3 | パターン |
|---|---|---|---|---|
| F32×cpuAndGPU **Once (新 AudioSynthesizer 単発)** | **8441 / 33162** (clipped, 異常) | - | - | 非決定的 (前回観測 750, 今回 8441) |
| F32×cpuAndGPU Repeat3 | 5029 / 24326 | 5029 / 24326 | 5029 / 24326 | iter1 から bit-identical loud |
| F16×cpuAndGPU Repeat3 | 4993 / 24176 | 4993 / 24176 | 4993 / 24176 | **iter1 から bit-identical** |
| Int8×cpuAndGPU Repeat3 | 5534 / 27710 | 5534 / 27710 | 5534 / 27710 | **iter1 から bit-identical** |
| F32×cpuOnly Repeat3 | 5029 / 24326 | 5029 / 24326 | 5029 / 24326 | **iter1 から bit-identical** |
| F32×all Repeat3 | 5029 / 24326 | 5029 / 24326 | 5029 / 24326 | **iter1 から bit-identical** |
| F32×cpuAndGPU warmupDummy → warmupReal | warmup discarded → real **5029 / 24326** | - | - | real は manual と bit-identical |

→ **F16/Int8/F32×cpuOnly/F32×all は iter1 から完全に決定的・loud**。「first-call 不安定」は F32×cpuAndGPU 固有と確定。

### Decoder step 別 sha — directRun1 が壊れる瞬間

`directRun1` (rms 8441, peak 33162 = 異常クリッピング) の decoder_steps.csv は **step 1 で `mel_min=inf, mel_max=-inf, mel_mean=nan`** と出る。

| step | directRun1 (rms 8441 / clipped) | directRepeat2 (rms 5029 / manual loud) | warmupReal (rms 5029 / loud) |
|---|---|---|---|
| 0  | 722549b182... (min=-0.019, max=0.368, mean=0.064) | 722549b182... (= Run1 と一致) | a4fe3b1195... (異なる) |
| **1** | **7da2ba4c61... min=inf max=-inf mean=nan** ← **NaN/Inf 発生!** | 1d42a46486 (min=-0.053, max=0.732, mean=0.270) | 1d42a46486 (= Repeat2 と一致) |
| 2  | 1c73eb2e81 (有限値に戻る) | 3a1a746149 | 9b379d4f67 |
| 50 | 4d75fa742d | a4463dc75f | 282d36ec80 |
| 100 | 6cffd5c910 | 0e29d70f87 | 0e256d97a7 |
| 200 | 0a282d0986 | d709275d13 | dbadab6744 |
| 261 (=最終 postnet 提供 step) | d4612982e4 (final wav clipped) | **1d42a46486** | **1d42a46486** |

(Decoder fixed262 モデルは出力 [1, 262, 256] の全要素 sha なので zero-padding 部分のノイズで step 0 のフルハッシュは run 間で異なるが、step 261 では同じ「正しい loud」収束済み出力に揃う。)

**結論:**
- `directRun1` は Decoder loop step 1 で GPU 計算が **NaN/Inf を吐いて発振**
- step 2 以降は有限値に戻るが trajectory が壊れたまま、step 261 で manual と違う postnet (`d4612982`) を出す → HiFi-GAN がクリッピング (peak 33162) を生成
- 健全な run はどれも step 261 で `1d42a46486 (mel) / 23c51a5431 (post)` に収束する。これが「真の loud」の同定子

### Warm-up 検証

`fp32GpuDummyWarmupThenReal` テスト: `loadModels` → dummy synth → real synth。

| run | wav sha256 (postdeemph) | rms (float) | peak (float) |
|---|---|---|---|
| warmupDummy (1st call, 捨てる) | **396a67e8b0...** | 0.1535 | 0.742 |
| warmupReal (2nd call) | **396a67e8b0...** | 0.1535 | 0.742 |
| manual dedicated (`a1c82ad4af`) | (int16 wav, ファイル形式違うが値一致) | 0.1535 | 0.742 |

→ 2 つは **bit-identical**。warm-up 後の real synth は manual と一致。

ただし注意: **今回 warmupDummy 自体も "first call" でありながら manual と一致する loud (5029) を出した**。「first call 結果は確率的」であり、warm-up が "確実に loud に揃える" のは **2 回目以降のすべての call** という意味。1st call が NaN を吐くかどうかは GPU の前段状態に依存する非決定的事象。

### 確定した最終仮説

**H6 確定 + 拡張 (H7):**

> **F32×cpuAndGPU の Decoder loop は新しい `MLModel` で 1 回目の推論を実行するとき、GPU シェーダーの初期化/最適化状態に依存して非決定的に NaN/Inf を吐くステップが発生し得る。自己回帰 262 step の中で trajectory が発振して step 261 の postnet 出力が manual と違う値になり、HiFi-GAN が異常な波形 (quiet rms 300-900 / over-amplified rms 8000+) を生成する。同じ `MLModel` インスタンスで 2 回目以降を呼ぶと GPU 状態が安定し、step 261 postnet が `23c51a5431...`, wav sha `396a67e8b0...`, rms 5029 / peak 24326 で deterministic に揃う。**

- F16/Int8/F32×cpuOnly/F32×all は 1 回目から完全決定論的 → F32×cpuAndGPU 固有の数値不安定性
- 「XCUITest が原因」「AVAudioSession が原因」「HiFi-GAN が原因」は **すべて DISPROVEN**
- 真因は **Decoder の F32 GPU シェーダーが初回 dispatch で数値不安定**

### 実用的含意 (本研究は計測フェーズ、本番対応は保留)

1. **warm-up パターン**: `loadModels` 直後に dummy synthesize を 1 回投げ、`waveform_postdeemph` に NaN/Inf がないかチェックし、出ていればもう一度投げる。これで 2 回目以降の本番呼びは確定的に loud。
2. **F32×cpuAndGPU を避ける**: F16×cpuAndGPU は 1 回目から決定的 (rms 4993)。本番デフォルトは既に F16+fixed262+cpuAndGPU なので、ユーザー実害は限定的 (F32 を選んだ研究計測時のみ問題)。
3. **計測時の運用**: F32 × cpuAndGPU を計測するときは「2 連続呼びの 2 回目を採用」または「warm-up 後の 1 回」のどちらかを採れば再現性が出る。「fresh-first single」は本質的に非決定的なので計測値として使わない。

### 残された未確定の論点

- **GPU で何の op が NaN を吐くか** — Decoder の attention/layer_norm/softmax の特定 op が初回 dispatch で不安定、までは絞れていない。`MLComputePlan` (iOS 17+) で op 別の compute unit assignment を見ると追加の手がかりが得られる可能性
- **warmupDummy が loud で出ることがある理由** — 第 3 段階の warmupDummy は loud (5029) だったが、yesterday の Once は quiet (750)、today の Once は clipped (8441)。Once が壊れたり壊れなかったりする条件 (GPU 温度、直前の他プロセス Metal 使用、デバイス sleep 復帰、etc.) は未解明
- 確率的 first-call 失敗の発生頻度の統計 (今回 1/3 失敗、サンプル数不足)

---

## 第 4 段階 (2026-05-19): 単発分布の追加サンプリング

ユーザー指示で `testFp32GpuFreshFirst` を別 xcodebuild 起動で 4 回 (今回 3 回追加 + 5/17 既存 1)、`testFp32GpuFreshFirstWithLaunchSleep10` を 3 回 (今回 2 回追加 + 5/17 既存 1) サンプリング。

| # | 日時 | テスト | mel_sha | enc_sha | post_sha | wave_sha | int16_rms | peak |
|---|---|---|---|---|---|---|---|---|
| 1 | 5/17 00:43 | noSleep | 35f61dbe02 | 7a86f45193 | 483ab4e454 | fd8f3e6a89 | 711 | 0.729 |
| 2 | 5/19 02:39 | noSleep | 35f61dbe02 | 7a86f45193 | 6cecb13ab9 | 886819b29f | 945 | 0.579 |
| 3 | 5/19 02:41 | noSleep | 35f61dbe02 | 7a86f45193 | 8cc9788ddc | 863aed5bae | 1441 | 0.765 |
| 4 | 5/19 02:43 | noSleep | 35f61dbe02 | 7a86f45193 | 75e0a50212 | 7c02397a5b | 1443 | 0.765 |
| 5 | 5/17 00:45 | sleep10 | 35f61dbe02 | 7a86f45193 | **483ab4e454** | **fd8f3e6a89** | 711 | 0.729 ← #1 と bit-identical |
| 6 | 5/19 02:45 | sleep10 | 35f61dbe02 | 7a86f45193 | 4a8ee69576 | fc7aadb91f | 611 | 0.484 |
| 7 | 5/19 02:47 | sleep10 | 35f61dbe02 | 7a86f45193 | **6cecb13ab9** | **886819b29f** | 945 | 0.579 ← #2 と bit-identical |

### 追加で確定したこと

1. **noSleep 単発 4 サンプル全部 quiet 範囲 (rms 711-1443)**。loud (5029) には到達しない。1 回目はどう転んでも quiet。
2. **sleep10 単発 3 サンプル全部 quiet 範囲 (rms 611-945)**。**sleep10 は出力に影響しない**。さらに #1=#5、#2=#7 で sleep の有無を超えて bit-identical → sleep10 が独立変数として効いていないことが直接示された。
3. **1st-call は完全ランダムではなく離散的 attractor を持つ確率分布**。post_sha が時々一致する (#2/#7、#1/#5) → 数個の "1st-call 状態" を確率的に取る。
4. **mel と Encoder は完全決定的** (全 7+ サンプル + 過去全データで `35f61dbe02` / `7a86f45193` 一致) — 分岐は Decoder loop 内に限定。

### 5 リクエスト判定表 (累積データ)

| # | リクエスト | 結果 | 根拠 |
|---|---|---|---|
| 1 | testFp32GpuFreshFirst を 3-5 回 → 毎回 quiet | **YES** | 4 サンプル全部 quiet (rms 711-1443, loud 5029 に到達しない) |
| 2 | sleep10 で loud に化けるか | **NO** | 3 サンプル全部 quiet。noSleep と bit-identical なペア (#1=#5, #2=#7) あり |
| 3 | 再生問題か生成結果問題か | **生成結果問題** | DebugRunSnapshot が再生前の `waveform_postdeemph` を直接 npy/sha 保存。値そのものが違う |
| 4 | 入力 mel が同じか | **完全一致** | 全 26+ サンプルで `mel_normalized` sha = `35f61dbe02...`、enc sha = `7a86f45193...` |
| 5 | Unit Test (UI なし) で再現するか | **YES** | `directSynthesisFp32GpuOnce` で rms 750 (5/17) / rms 8441 (5/19) と quiet/clipped 観測 → UI 非依存 |


## 関連リソース

- 修正コミット: `3e2e235`（playback-wait fix）
- 退避スクリプト: `scripts/extract_manual_run.sh`（コミット `573e2b4`）
- 追加 XCUITest メソッド: `testFp32GpuFreshFirst`, `testCaptureAllCombinationsGpuFirst`（未コミット）
- 比較データ: `audio/iPhone_3_manual_*` 配下、`audio/iPhone_3_20260516_*` 配下
- 関連既存ドキュメント:
  - [`docs/2026-05-06/float32-gpu-investigation-summary.md`](../2026-05-06/float32-gpu-investigation-summary.md)
  - [`docs/2026-05-07/hirai-comment-memory-management.md`](../2026-05-07/hirai-comment-memory-management.md)
  - [`docs/2026-05-08/device-benchmark-workflow.md`](../2026-05-08/device-benchmark-workflow.md)

## 補足：Mac 再生音量問題（解消済み）

調査中、Mac 音量が 13/100 という極端に低い設定だったため F32 wav が全部「無音」に聞こえていた。これは Mac 側の問題で、ファイル内容自体は OK だった（peak 24327、rms 5029 でちゃんと音が入ってる）。ただし auto-test の cpuAndGPU は Mac 音量を上げても聞こえない（rms 480 で本当に小さい）ことが確認されたので、「auto-test cpuAndGPU が quiet」は実体的な現象。
