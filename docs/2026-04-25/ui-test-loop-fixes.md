# XCUITest 12通りループのフレーク対策3点

`feature/ui-test-combinations` ブランチで `testCaptureAllCombinations`
（精度3 × 計算デバイス4 = 12通りを順番に回すテスト）を信頼できる形に
整えるために入れた修正の解説。

XCUITest は通常「1テスト = 1シナリオ」で書く前提だが、本テストは
1メソッドのなかで12組み合わせを順番に回しているので、**前回の合成結果が
UI に残っている状態で次の操作を始める**ことになる。
これが原因で発生する3つの落とし穴を潰した、というのが今回の作業。

| # | 落とし穴 | 修正 | コミット |
|---|---|---|---|
| 1 | 失敗が起きるとカウントが2倍になる | ループ内の `XCTFail` 削除 | `a277479` |
| 2 | 直前の合成完了直後は picker がまだ disabled でタップが空振りする | picker 操作前に `isEnabled` を待つ | `4c44a14` |
| 3 | 直前のステータス "合成完了" に即マッチして合成を待たずに進む | ボタンの `isEnabled` 遷移で待つ | `d3f46e6` |

3つとも本質は「**前回の状態が残っているうちに次の判定をしない**」こと。

---

## 修正 1: ループ内の `XCTFail` を削除（`a277479`）

### 何が起きていたか

```swift
continueAfterFailure = true   // 1件失敗しても残り11件続行したい
...
catch {
    failures.append(...)
    XCTFail("組み合わせ失敗 ...")     // ← ここで1回
}
...
if !failures.isEmpty {
    XCTFail("失敗した組み合わせ: \(failures.count) 件...")  // ← ループ後にもう1回
}
```

`continueAfterFailure = true` だとテストを最後まで続けるので、両方の `XCTFail`
が実行される。3件失敗したのに「6件失敗」とレポートされてしまう。

### 修正

ループ内の `XCTFail` を消し、最後の集約用 `XCTFail` だけ残す。
失敗の詳細は `failures` 配列に積んでおき、最後にまとめて1回だけ報告する。

これは純粋な「報告ロジックの整理」で、テスト動作自体は変わらない。

---

## 修正 2: picker 操作前に `isEnabled` を待つ（`4c44a14`）

### 何が起きていたか

ContentView 側では合成中、picker・ボタンを `.disabled(viewModel.isProcessing)`
で無効化している。`isProcessing` は ViewModel の `runSynthesis` が完了するまで `true`。

ところが ViewModel の終了処理はこの順で走る:

```swift
status = "合成完了"   // ← UI に通知（ここでテスト側が「終わった」と判定する）
progress = 1.0
playOutput()
return
defer { isProcessing = false }   // ← defer は関数を抜ける時点で実行
```

つまり「ステータスが 合成完了 になった瞬間」と「picker が enabled に戻る瞬間」
のあいだに**短い窓**が空く。テスト側はステータスを見て次の組み合わせに進むので、
このタイミングで picker をタップしようとすると、まだ disabled で空振りする。

### 修正

`waitUntilEnabled` ヘルパーを追加して、`selectPrecision` / `selectComputeUnit`
の冒頭で `isEnabled == true` を待つ。

```swift
private func waitUntilEnabled(_ element, ...) {
    if element.isEnabled { return }    // 即 enabled なら何もしない
    expectation(for: NSPredicate(format: "isEnabled == true"), ...)
    waitForExpectations(timeout: 10)
}
```

XCTest の `expectation(for:evaluatedWith:)` は、「指定した predicate が true に
なったら fulfill する」というポーリングの仕組み。自分で `while !element.isEnabled`
を書かなくていい。

---

## 修正 3: 合成完了の待機をボタン遷移で取る（`d3f46e6`）

これは Codex のレビューが指摘してくれた、一番ミクロな問題。

### 何が起きていたか

修正前の合成完了待ちはこう書いていた:

```swift
button.tap()   // 合成開始を指示
let done = NSPredicate(format: "label == %@ OR label == %@", "合成完了", "エラー")
expectation(for: done, evaluatedWith: statusLabel, ...)
waitForExpectations(timeout: 180)
```

シンプルに「ステータスが 合成完了 になるまで待つ」書き方。1回目はうまくいく。
問題は **2回目以降のループ**。

時系列で見ると:

```
[1回目の合成終了] status = "合成完了"
                   ↓
[次の組み合わせ選択]  selectPrecision, selectComputeUnit
                   ↓
button.tap()        ← 2回目の合成開始を「指示」しただけ
                   ↓
                   この瞬間 status はまだ "合成完了" のまま。
                   ViewModel の runSynthesis は async なので、
                   Task → @MainActor へのホップが走るまで何も書き換わらない。
                   ↓
expectation(for: done, ...) を登録
                   ↓
waitForExpectations が即座に最初の評価をする
                   ↓
status == "合成完了" → 即マッチ！
                   ↓
合成本体を待たずに次のループへ
```

タイミングが噛み合うと、合成を待たずに次に進んでしまう。
出力ファイル `output_mel_<P>_<U>.npy` は組み合わせごとに固定名で書き出すので、
ファイル名は12通り並ぶが**中身が前回の合成結果のまま**、というサイレントな
破綻が起きる可能性があった（テストは "passed" になる）。

### 修正

ステータスのラベル（文字列）ではなく、合成ボタンの `isEnabled` の**遷移**を見る。

```swift
button.tap()

// ① 合成が始まったことの確認: ボタンが disabled になる
let processingPred = NSPredicate(format: "isEnabled == false")
expectation(for: processingPred, evaluatedWith: button, ...)
waitForExpectations(timeout: 5)

// ② 合成が終わったことの確認: ボタンが enabled に戻る
let finishedPred = NSPredicate(format: "isEnabled == true")
expectation(for: finishedPred, evaluatedWith: button, ...)
waitForExpectations(timeout: 180)

// ③ 終わった後にラベルを見て成功/失敗を判定
let finalStatus = statusLabel.label
if finalStatus != "合成完了" {
    throw UITestError.synthesisDidNotComplete(status: finalStatus)
}
```

なぜこれで直るか:

- `button.isEnabled` は `viewModel.isProcessing` に直結している
- `isProcessing` は **必ず `false → true → false`** と遷移する
  （`runSynthesis` の最初で `true`、`defer` で `false`）
- 前回の状態に左右されず、必ず disabled 期間を挟むので「合成を待った」ことが確定する

ラベル監視と違うのは「観測する値の更新タイミング」:

| 観測対象 | 更新タイミング | 残留リスク |
|---|---|---|
| `status` ラベル | ViewModel が文字列を代入したときだけ | **あり**（書き換え前は前回値が残る） |
| `button.isEnabled` | `isProcessing` の変化に SwiftUI が即追従 | なし（state 駆動の 1bit） |

---

## まとめ表

| 修正 | 観測対象 | 待ち方 | 効果 |
|---|---|---|---|
| 1 | （報告ロジックの整理） | — | 失敗件数が正しく出る |
| 2 | picker `isEnabled` | `true` になるまで | 次の組み合わせ選択が空振りしない |
| 3 | synthesizeButton `isEnabled` | `false → true` の2段 | 合成を待たずに進むことがない |

---

## 関連ファイル

- `ios/CoreMLAudioApp/CoreMLAudioAppUITests/CoreMLAudioAppUITests.swift` — 修正対象
- `ios/CoreMLAudioApp/CoreMLAudioApp/Views/ContentView.swift` — `.disabled(viewModel.isProcessing)` の元
- `ios/CoreMLAudioApp/CoreMLAudioApp/ViewModels/SynthesisViewModel.swift` — `isProcessing` / `status` の制御
- PR #13: https://github.com/orukahairuka/coreml-audio-research/pull/13
