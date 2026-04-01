# iOS アプリ (CoreMLAudioApp) 実装詳細

`feature/ios-app` ブランチで実装した iOS アプリの詳細をまとめる。

## 概要

PronounSE の音声合成パイプライン（Python 版 `synthesis.py`）を iOS 上で再現するアプリ。
CoreML モデル 3 つ + Swift ネイティブの前処理で構成される。

## 全体パイプライン

```
入力音声 (.wav)
  │
  ├─ [CPU / Accelerate] AudioFeatureExtractor
  │   ├─ WAV 読み込み・22050Hz リサンプリング
  │   ├─ オンセット検出 + フェードアウト
  │   ├─ プリエンファシスフィルタ
  │   ├─ STFT (vDSP_fft_zrip)
  │   ├─ メルフィルタバンク適用
  │   └─ dB 正規化
  │   → メルスペクトログラム [T, 256]
  │
  ├─ [CoreML / CPU+GPU] Transformer Encoder
  │   入力: mel [1, T, 256], pos [1, T]
  │   出力: memory (Decoder へのコンテキスト)
  │
  ├─ [CoreML / CPU+GPU] Transformer Decoder (自己回帰ループ × T 回)
  │   入力: memory, decoder_input [1, t, 256], pos [1, t]
  │   出力: mel_out, postnet_out (精製メルスペクトログラム)
  │
  ├─ [CoreML / CPU+GPU] HiFi-GAN Vocoder
  │   入力: mel [1, 256, T] (転置済み)
  │   出力: waveform (生波形)
  │
  └─ デエンファシスフィルタ
      y[n] = x[n] + 0.97 * y[n-1]
      → 最終波形 → AVAudioPlayer で再生
```

---

## ファイル構成

```
ios/CoreMLAudioApp/CoreMLAudioApp/
├── CoreMLAudioAppApp.swift       # アプリエントリーポイント
├── ContentView.swift             # SwiftUI UI
├── AudioFeatureExtractor.swift   # 音声前処理（メルスペクトログラム抽出）
├── AudioSynthesizer.swift        # CoreML 推論パイプライン
└── input_sample.wav              # テスト用入力音声
```

---

## 1. AudioFeatureExtractor.swift

音声ファイルからメルスペクトログラムを抽出する。Python 版 `PronounSE/Transformer/utils.py` の `get_spectrograms` に相当。

### ハイパーパラメータ

| パラメータ | 値 | 説明 |
|---|---|---|
| `sampleRate` | 22050 Hz | サンプリングレート |
| `nFFT` | 1024 | FFT サイズ |
| `hopLength` | 256 | STFT のホップ長 |
| `winLength` | 1024 | 窓関数の長さ |
| `nMels` | 256 | メル周波数ビン数 |
| `maxDB` | 100 | dB 正規化の最大値 |
| `refDB` | 20 | dB 正規化のリファレンス値 |
| `preemphasisCoeff` | 0.97 | プリエンファシス係数 |

すべて `PronounSE/Transformer/hyperparams.py` と一致させている。

### 処理フロー

#### (1) loadAudio(from:)

- `AVAudioFile` で WAV ファイルを読み込む
- 元のサンプルレートが 22050 Hz でない場合は `AVAudioConverter` でリサンプリング
- モノラルの Float32 配列として返す

#### (2) preprocess(_:)

- **オンセット検出**: エネルギー閾値 (0.08) を超える最初の窓位置を検出し、それ以前の無音をトリミング
  - 窓幅 1024 サンプル、シフト 256 サンプルで走査
  - 各窓内の二乗和がしきい値を超えたらその位置を onset とする
- **フェードアウト**: 末尾 15% の区間に線形フェードアウトを適用

#### (3) applyPreemphasis(_:)

- 高周波を強調するフィルタ
- `y[n] = x[n] - 0.97 * x[n-1]`
- 音声信号は低周波成分が支配的なため、前処理で高周波を持ち上げることでモデルの学習・推論精度を上げる

#### (4) stft(_:)

STFT（短時間フーリエ変換）を実行し、magnitude spectrogram `[T, 513]` を返す。

- Hann 窓を適用して短い区間を切り出す
- Accelerate の `vDSP_fft_zrip` で FFT を実行
- FFT 結果から magnitude（振幅スペクトル）を計算:
  - DC 成分: `|real[0]| / N`
  - ナイキスト成分: `|imag[0]| / N`
  - その他: `sqrt(re² + im²) * 2 / N`
- パディング処理: 信号長が FFT サイズに満たない場合はゼロパディング

```swift
for frame in 0..<frameCount {
    let start = frame * hopLength
    // 窓関数を適用して切り出し
    // vDSP_fft_zrip で FFT 実行
    // magnitude を計算・保存
}
```

#### (5) applyMelFilterbank(_:)

- `createMelFilterbank()` でメルフィルタバンク行列 `[256, 513]` を生成
  - Hz ↔ Mel 変換: `mel = 2595 * log10(1 + hz/700)`
  - 256+2 個の等間隔メル周波数点を計算
  - 三角フィルタを構築
  - Slaney 正規化（帯域幅で割る）
- 行列積で magnitude spectrogram `[T, 513]` → メルスペクトログラム `[T, 256]` に変換

#### (6) normalizeToDB(_:)

- dB 変換: `20 * log10(max(1e-5, value))`
- 正規化: `(dB - 20 + 100) / 100`
- クリッピング: `[1e-8, 1.0]` の範囲に制限

---

## 2. AudioSynthesizer.swift

3 つの CoreML モデルを使って音声合成を実行する。`@Observable` で UI にステータスと進捗を公開。

### クラス設計

```swift
@MainActor
@Observable
final class AudioSynthesizer {
    var status: String       // 現在の処理ステータス（UI 表示用）
    var isProcessing: Bool   // 処理中フラグ（ボタン制御用）
    var progress: Double     // 進捗率 0.0〜1.0（プログレスバー用）

    private var encoder: MLModel?
    private var decoder: MLModel?
    private var hifigan: MLModel?
}
```

### loadModels()

- `Bundle.main` から 3 つの `.mlmodelc` をロード
  - `Transformer_Encoder.mlmodelc`
  - `Transformer_Decoder.mlmodelc`
  - `HiFiGAN_Generator.mlmodelc`
- `MLModelConfiguration.computeUnits = .cpuAndGPU` で CPU + GPU のハイブリッド実行
- 2 回目以降の呼び出しはスキップ（既にロード済みの場合）

### synthesize(inputURL:) async throws -> [Float]

5 ステップで合成を実行する。

#### ステップ 1: 特徴量抽出

- `AudioFeatureExtractor.extractMelSpectrogram()` を呼び出し
- メルスペクトログラム `[T, 256]` とフレーム数 T を取得

#### ステップ 2: Encoder

- 入力:
  - `mel`: `MLMultiArray [1, T, 256]` (Float32) — メルスペクトログラム
  - `pos`: `MLMultiArray [1, T]` (Int32) — 位置エンコーディング `[1, 2, ..., T]`
- 出力:
  - `memory` — Encoder の隠れ状態（Decoder のクロスアテンションで参照される）
- 1 回のフォワードパスで完了

#### ステップ 3: Decoder（自己回帰ループ）

T 回のループで逐次的にメルスペクトログラムを生成する。最も時間のかかる処理。

- 初期入力: ゼロベクトル `[1, 1, 256]`
- 各ステップで:
  - 入力:
    - `memory`: Encoder 出力（固定）
    - `decoder_input`: `MLMultiArray [1, t, 256]` — これまでの出力を蓄積した系列
    - `pos`: `MLMultiArray [1, t]` — 位置エンコーディング
  - 出力:
    - `mel_out`: 生のメルスペクトログラム出力
    - `postnet_out`: PostNet で精製されたメルスペクトログラム
  - `mel_out` の最後のフレームを `decoder_input` に追加して次ステップへ
- 10 ステップごとに `Task.yield()` で UI をブロックしない

#### ステップ 4: HiFi-GAN Vocoder

- 入力の転置: `postnet_out [1, T, 256]` → `[1, 256, T]`（チャネルファースト）
- HiFi-GAN でメルスペクトログラム → 波形に変換
- 出力は 22050 Hz の Float32 波形

#### ステップ 5: デエンファシスフィルタ

- プリエンファシスの逆操作: `y[n] = x[n] + 0.97 * y[n-1]`
- 前処理で強調した高周波を元のバランスに戻す

### エラーハンドリング

| エラー | 内容 |
|---|---|
| `modelNotFound` | `.mlmodelc` がバンドルに存在しない |
| `modelNotLoaded` | `loadModels()` 未実行で `synthesize()` を呼んだ |
| `decoderFailed` | Decoder の出力取得に失敗 |

---

## 3. ContentView.swift

SwiftUI で構築されたメイン UI。

### 状態管理

```swift
@State private var synthesizer = AudioSynthesizer()  // 合成エンジン
@State private var audioPlayer: AVAudioPlayer?        // 音声再生
@State private var errorMessage: String?              // エラー表示
@State private var outputWaveform: [Float]?           // 合成結果
```

### UI 構成

```
NavigationStack "CoreML Audio"
├── GroupBox "ステータス"
│   ├── ステータステキスト
│   └── ProgressView（処理中のみ表示）
├── ボタン群
│   ├── [合成実行]  — 処理中は無効化
│   ├── [再生]      — 合成結果がないか処理中は無効化
│   └── [停止]      — 再生中でなければ無効化
├── エラーメッセージ（赤字、エラー時のみ表示）
└── GroupBox "情報"
    ├── モデル: PronounSE (Float16)
    ├── サンプルレート: 22050 Hz
    └── 入力: input_sample.wav (バンドル)
```

### runSynthesis()

1. `synthesizer.loadModels()` でモデルロード
2. バンドルから `input_sample.wav` の URL を取得
3. `synthesizer.synthesize(inputURL:)` で合成実行
4. 結果を `outputWaveform` に保存
5. `playOutput()` で自動再生

### playOutput()

1. iOS ではオーディオセッションを `.playback` に設定
2. Float32 波形 → `AVAudioPCMBuffer` に格納
3. 一時ディレクトリに Int16 PCM WAV ファイルとして書き出し
4. `AVAudioPlayer` で再生
5. `audioPlayer` を `@State` に保持（GC 防止）

### stopPlayback()

- `audioPlayer?.stop()` で再生停止

---

## 4. CoreMLAudioAppApp.swift

```swift
@main
struct CoreMLAudioAppApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
```

標準的な SwiftUI アプリエントリーポイント。特別なロジックなし。

---

## 使用フレームワーク

| フレームワーク | 実行先 | 用途 |
|---|---|---|
| Accelerate (vDSP) | CPU (SIMD/NEON) | FFT、窓関数、magnitude 計算 |
| AVFoundation | CPU | WAV 読み込み、リサンプリング、再生 |
| CoreML | CPU + GPU | Encoder / Decoder / HiFi-GAN 推論 |
| SwiftUI + Observation | - | UI・状態管理 |

---

## Python 版との対応関係

| Python (PronounSE) | Swift (CoreMLAudioApp) |
|---|---|
| `Transformer/utils.py` `get_spectrograms()` | `AudioFeatureExtractor.extractMelSpectrogram()` |
| `Transformer/utils.py` `preprocess()` | `AudioFeatureExtractor.preprocess()` |
| `librosa.filters.mel()` | `AudioFeatureExtractor.createMelFilterbank()` |
| `synthesis.py` encoder 実行 | `AudioSynthesizer.synthesize()` ステップ 2 |
| `synthesis.py` decoder ループ | `AudioSynthesizer.synthesize()` ステップ 3 |
| `synthesis.py` HiFi-GAN 実行 | `AudioSynthesizer.synthesize()` ステップ 4 |
| `scipy.signal.lfilter` (de-emphasis) | `AudioSynthesizer.synthesize()` ステップ 5 |
