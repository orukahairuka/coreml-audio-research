# PANNs による FAD の評価環境

参照効果音と合成音（NVIDIA GPU・FP32、iPhone 13、iPhone 17 Pro）の分布の近さを、
PANNs の埋め込みによる FAD（Fréchet Audio Distance）で測るための環境。
iOS アプリ（`ios/`）と推論・計測のコード（`scripts/`、`PronounSE/`）からは独立しており、
それらを import も変更もしない。入力は wav のフォルダだけ。

## 採用する方法と前処理

| 項目 | 内容 | 理由 |
|---|---|---|
| 埋め込み | PANNs Wavegram-Logmel-CNN14（AudioSet、mAP 0.439）。2048 次元、クリップごとに 1 つ、float32 | DCASE 2024 Task 7（効果音・環境音の生成）の公式評価のモデル。主観評価との相関が CLAP・VGGish より高かった |
| 重み | `Wavegram_Logmel_Cnn14_mAP=0.439.pth`（Zenodo 3987831、SHA-256 `6c1c917d…`） | kadtk 1.1.0 の `PANNsModel` がモデルの定義と取得を行う |
| FAD | ‖μ_r − μ_e‖² + Tr(Σ_r) + Tr(Σ_e) − 2 Tr((Σ_r Σ_e)^{1/2})、不偏共分散、float64 | 元の式どおり。平均の項と共分散の項を分けて出す |
| Tr((Σ_r Σ_e)^{1/2}) | 中心化した埋め込みの行列 M = X_c Y_cᵀ の特異値の和 / √((n_r−1)(n_e−1)) | 元の式と数学的に等しい（下記）。音が少ないとき sqrtm より誤差が出にくい |
| 標本化周波数 | 読み込み → 22,050 Hz（帯域をそろえる）→ 32 kHz（PANNs の入力）。soxr_hq | PronounSE の出力は 22,050 Hz。参照音の 11 kHz 以上の成分の有無で差が出ないようにする |
| チャンネル | 平均してモノラル | |
| 入力の形式 | FAD には 16 ビットの wav を使う（±1 を超えた部分は切られた状態）。クリッピング前の振幅の異常を調べるときは、FAD とは別に float32 の npy（iPhone は `*_deemph.npy`）を使う | 聞こえる音（保存される wav）を評価する |
| 振幅 | 正規化しない | 振幅の違いも評価に含める |
| 長さ（合成音どうし） | 同じ名前（同じ入力）の音を、合成音の集合の中で最も短い長さに先頭から切る。0 埋めはしない | PANNs は時間方向にプーリングするので、長さの違いが FAD に入らないようにする |
| 長さ（参照音） | **未確定**。実際の参照音が届いてから決める（下の「長さの扱い」） | 参照音と合成音が時間で対応しているかが分からないため |
| 標本 | 1 つの入力から 1 つの音。同じ集合に同じ波形があれば止める | 同じ入力の繰り返しは独立な標本ではない |

### 等価性

Σx = X_cᵀX_c/(n_x−1)、Σy = Y_cᵀY_c/(n_y−1) とする。Σx Σy は半正定値行列の積なので固有値は実数で 0 以上であり、
Tr((Σx Σy)^{1/2}) = Σ_i √λ_i(Σx Σy)。AB と BA の 0 でない固有値は等しいので、
λ(Σx Σy) の 0 でないものは λ(M Mᵀ)/((n_x−1)(n_y−1)) = σ(M)²/((n_x−1)(n_y−1))。
よって Tr((Σx Σy)^{1/2}) = Σ_i σ_i(M) / √((n_x−1)(n_y−1))。
テスト（`test_compute_fad.py`）で、正則な場合は sqrtm による元の式と相対 1e-9 で、
正則でない場合（n = 4、24、50 / 次元 2048）は float64 の kadtk と相対 1e-6 で一致することを確かめた。

### kadtk の値とのずれ（約 0.2）

同じ埋め込みを入れて 1 つずつ条件を変えて調べた結果、原因は kadtk が共分散の積とその sqrtm を
**float32** で計算することだった。共分散が正則でないと float32 の sqrtm の誤差が大きくなる。

| 計算 | same | clipped |
|---|---|---|
| 本実装（float64、特異値） | 8.4883 | 12.2048 |
| kadtk（precision=float64） | 8.4883 | 12.2048 |
| 共分散は float32、sqrtm は float64 | 8.4883 | 12.2048 |
| kadtk（既定の float32） | 8.3057 | 12.0089 |
| 埋め込みを float16 に丸めた後の本実装 | 8.4883 | 12.2047 |

埋め込みは同じで、float16 への丸めの影響は 1e-3 以下、共分散の float32 の影響は 1e-5 以下。

## 長さの扱い

- NVIDIA 側（公開実装）は T フレーム全体、iPhone 側は 256T′ サンプル（T′ = min(T, 262)）を出す。
  T ≤ 262 の入力では長さは同じで、T > 262 の入力（入力 1・6）では NVIDIA 側の後ろを切ることになる
- 参照音の長さの扱いは未確定。実際の参照音の長さと、合成音との時間の対応を見てから、次のどちらにするかを決める
  - 参照音も同じ長さに切る（`--align shortest-by-name` の既定の動作）
  - 参照音は切らず、合成音どうしだけそろえる（`--keep-ref-length`）
  決めるまでは、両方の結果を出して比べる
- `--align fixed --crop-sec S` はすべてを S 秒に切る。`--align none` は切らない

## 音の数が少ない場合

FAD は音の数 n に強く依存する（小さいほど大きく、ばらつく）。テストデータ（24 組から n 組を 300 回選んだ平均 ± 標準偏差）:

| 集合 | n = 4 | n = 8 | n = 16 | n = 24 |
|---|---|---|---|---|
| same | 33.3 ± 10.6 | 21.8 ± 5.0 | 12.4 ± 1.6 | 8.5 |
| clipped | 41.3 ± 12.6 | 26.5 ± 5.7 | 16.6 ± 1.9 | 12.2 |
| noisy | 169.4 ± 13.8 | 163.3 ± 9.0 | 159.9 ± 4.4 | 158.7 |

n = 4 では、same と clipped の大小が 8% の選び方で逆転した。また、4 個の音を 6 回ずつ複製して 24 個と数えると、
FAD は 27.6 から 23.1 に下がった（共分散の分母が変わるだけで、情報は増えていない）。

## 環境の作り方

```bash
cd eval/fad
uv venv --managed-python --python 3.11 .venv      # Homebrew の Python は tkinter が無く kadtk の import に失敗する
uv pip install --python .venv/bin/python -r requirements.lock.txt --no-deps
```

- kadtk は依存に TensorFlow・CLAP などを含むので、`--no-deps` で PANNs に必要なものだけを入れている
- `kadtk` の CLI と `get_all_models()` は使わない（CLAP の重み約 2.5 GB をダウンロードし、入力のフォルダに
  変換後の音やキャッシュを書き込むため）
- PANNs の重み（約 330 MB）は初回に `.venv/.../kadtk/.model-checkpoints/` にダウンロードされる
- 重みは CUDA 上で保存されているので、読み込みの間だけ `torch.load` に `map_location="cpu"` を付けている

## 使い方

```bash
.venv/bin/python compute_fad.py --ref data/reference \
    --eval nvidia_fp32=data/nvidia --eval iphone13=data/iphone13 --eval iphone17pro=data/iphone17pro \
    --save-embeddings embeddings/<名前> --out results/<名前>.json
```

- 各フォルダには、入力ごとに 1 つの wav を同じファイル名（例 `1.wav`、`6.wav`、`2.wav`、`11.wav`）で置く
- iPhone の音は計測モードが保存した `infer_<条件>_in<入力>_r<回>_deemph.wav` から、入力ごとに 1 回分
  （例: 最初の定常の回 r1）を選んでコピーする。同じ条件の繰り返しは同じ波形になる（2026-11-21 の実機の確認で
  SHA-256 が一致）ので、複数入れると止まる
- NVIDIA 側も 16 ビットの wav を使う（iPhone 側と同じく保存される音）
- 出力: 標準出力に FAD と内訳、照合用の sqrtm（float64）と kadtk（float32）の値。`--out` の JSON に方法・環境・
  切った長さ・ファイルごとの長さと振幅

## テスト

```bash
.venv/bin/python -m pytest -q                      # 式の等価性、前処理、長さ、標本の独立性（重みは使わない）
.venv/bin/python make_test_data.py /tmp/fadtest     # 合成音のテストデータ（参照 44.1 kHz ステレオ）
.venv/bin/python compute_fad.py --ref /tmp/fadtest/ref --eval ref_copy=/tmp/fadtest/ref_copy \
    --eval same=/tmp/fadtest/same --eval noisy=/tmp/fadtest/noisy --eval clipped=/tmp/fadtest/clipped
```

2026-11-21 の確認: pytest 11 件が通過。テストデータ（各 24 個、CPU、約 30 秒）で ref_copy 0.148 < same 8.49 <
clipped 12.20 < noisy 158.74、2 回の実行の値は完全に一致。

## 制約

- 音の数が 2048 以下では共分散が正則でなく、FAD は音の数に依存する。値は同じ参照音の集合・同じ入力の組・
  同じ前処理どうしでだけ比べ、他の研究の FAD の値とは比べない
- NVIDIA との比較には、計算環境の違いと固定長化（0 埋め・262 フレームでの打ち切り）の影響が混ざる

## 提供された評価データセット（2026-11-22 受領）

PronounSE の開発者から受け取った。再配布しないので Git には入れない（ルートの `.gitignore` で除外）。
2026-11-22 時点では `~/Desktop/` に置いてあり、リポジトリの中には無い。

| フォルダ | 中身 | 件数 | 形式 |
|---|---|---|---|
| `Trimmed_Explosion/` | 参照効果音 `<参照>.wav` | 100 | 96 kHz・モノラル・WAV PCM_24、0.62〜8.51 秒（中央 3.43） |
| `Vocal_Mimicry/<話者>/` | 口真似音 `<参照>_<話者>.wav`（f-01、m-00〜m-05） | 7 × 100 | 48 kHz・モノラル・WAV PCM_24、0.55〜8.53 秒 |
| `PronounSE/<話者>/` | オリジナル PronounSE の合成音 `<参照>_<話者>_pronounse.wav`（f-01、m-01〜m-05） | 6 × 100 | 22,050 Hz・モノラル・WAV PCM_16、0.56〜8.53 秒 |

- FAD の評価は f-01、m-01〜m-05 の 6 話者。m-00 は学習話者なので FAD に使わず、RTF の計測に使う
- 検証: `.venv/bin/python check_dataset.py <置き場所>`（読むだけ）。件数・名前の対応・命名規則・話者の一致・
  中身の重複（SHA-256）・WAV のメタデータを確かめる。2026-11-22 は問題 0 件
- 合成音の長さは口真似音とほぼ同じ（比 1.000〜1.020）で、オリジナル PronounSE は入力の全体を合成している。
  参照音との長さの比は 0.55〜1.29 で、合成音と参照音は時間で対応していない
- 口真似音のうち 262 フレーム（3.04 秒）を超えるものは、評価 6 話者で 53〜58 件 / 100（m-00 は 57 件）

### 開発者の FAD（PANNs-FAD、オリジナル PronounSE）

| 話者 | m-01 | m-02 | m-03 | m-04 | m-05 | f-01 | Whole |
|---|---:|---:|---:|---:|---:|---:|---:|
| FAD | 17.85 | 17.05 | 17.28 | 16.65 | 19.29 | 19.28 | 14.09 |

話者別は合成音 100 と参照音 100、Whole は合成音 600（6 話者）と参照音 100 を比べた値。原則として再計算しない。

### 開発者の計算条件と本実装の一致（2026-11-22 時点で確認できていないこと）

本実装と同じ条件であることは、まだ確かめられていない。次を開発者に確認する。

1. PANNs のモデル（CNN14 32k / 16k / Wavegram-Logmel-CNN14）と重み、埋め込みの層（2048 次元の embedding か）
2. 使った FAD の実装（fadtk / kadtk / frechet_audio_distance / 自作）と、統計の精度（float32 か float64 か）
3. 参照音（96 kHz）と合成音（22,050 Hz）の標本化周波数の変換（帯域を 22,050 Hz にそろえたか、32 kHz に直接か）
4. 長さの扱い（全体を入れたか、切ったか・0 で埋めたか、10 秒などの区間に分けて平均したか）
5. 振幅の正規化の有無、共分散の分母（n−1 か n か）

本実装との違いが分かっているもの:

- 本実装は 22,050 Hz を通すので、参照音の 11 kHz 以上は落ちる。開発者が 96 kHz → 32 kHz に直接変えていれば値は変わる
- kadtk の既定（float32）は本実装（float64）より小さく出る（ランダムな埋め込みで 100 対 600 のとき 0.13%、
  テストデータで約 0.2）
- `compute_fad.py` の `--align shortest-by-name` は全集合でファイル名が同じことを前提にしており、
  参照音（`<参照>`）と合成音（`<参照>_<話者>_pronounse`）の名前の違いには対応していない。
  また、`--eval` のフォルダは直下の wav だけを読むので、Whole（6 話者のフォルダにまたがる 600 件）はそのままでは渡せない
- 式は 100 対 100、100 対 600 でも元の式（sqrtm、float64）・kadtk（float64）と相対 1e-7 以内で一致した（2026-11-22）

## 正式評価の設定（2026-11-22 整理）

オリジナル PronounSE（開発者が提供した合成音）と、同じ入力の全体・同じ前処理・同じ後処理で比べる。

| 項目 | 設定 | 根拠 |
|---|---|---|
| モデル | `range1`（Encoder・Decoder・HiFi-GAN とも RangeDim 1〜1000）。固定長の `fixed262` は残すが正式評価には使わない | 入力を切らない。1000 フレーム ≒ 11.6 秒、データセットは最大 735 フレーム |
| 入力 | 口真似音を `librosa.load(path, sr=22050)` で 22050 Hz・モノラルにし、float32 の wav にしたもの（データセットとは別の場所に作る） | iOS は 48 kHz を AVAudioConverter で変換し、librosa（soxr_hq）と合わない。その差が自己回帰で広がり、波形が別物になる（下記） |
| 波形の前処理 | `AudioFeatureExtractor.WaveformPreprocess.fullInput`（先頭の無音を削らない・フェードしない） | 提供された合成音 600 件すべてで T = 1 + floor(N/256)。PyTorch で再合成してオリジナルと比べると、削らない方が近い（onset が 0 でない 9 件すべて）、フェードしない方が近い（フェードでメルが変わる 12 件中 11 件。残る 1 件はほぼ同じ）。`scripts/identify_original_preprocess.py` |
| メル | プリエンファシス 0.97 → STFT（n_fft 1024、hop 256、hann、center、0 埋め）→ メル 256（slaney）→ 20 log10(max(1e-5, ·)) → (x − 20 + 100) / 100 を [1e-8, 1] に切る | `get_spectrograms` と同じ。振幅の正規化はしない |
| Decoder | 0 の 1 フレームから始め、T 回まわす | `synthesis.py` と同じ |
| 後処理 | HiFi-GAN → デエンファシス y[n] = x[n] + 0.97 y[n−1] → PCM_16（±1 で切る）。フェード・正規化はしない | `synthesis.py` と同じ。iOS のデエンファシスは float32（差は 0.03 LSB 以下）、PCM_16 への丸めは soundfile と最大 1 LSB 違う。完全にそろえるなら iOS の float32 の波形（npy）をホストで soundfile により PCM_16 にする |
| 出力の長さ | T × 256 サンプル | 提供された合成音と同じ |
| 音質評価（FAD） | f-01、m-01〜m-05 の各 100 件（計 600）と参照音 100 件。話者別 6 条件 + Whole | 開発者の値と同じ 7 条件 |
| RTF 評価 | m-00 の 100 件 | 学習話者なので FAD には使わない |

### 前処理の同等性（2026-11-22、4 サンプル、Mac で iOS の前処理のコードをビルドして確認）

| 入力の経路 | 読み込み後の波形 | メル max\|差\| | T | 合成波形（Core ML range1 F32 CPU ↔ PyTorch、同じ条件） |
|---|---|---|---|---|
| 22050 Hz の float32 wav | librosa と完全に一致 | 3.7e-6 以下 | 4 件とも一致 | SNR 34〜67 dB、cos ≥ 0.9998 |
| 48 kHz の wav（AVAudioConverter） | 1 サンプル短く、値も違う | 0.28〜0.37 | 4 件とも一致 | SNR −4〜1 dB、cos 0.02〜0.54（別の音） |

- 検証: `scripts/verify_fullinput_preprocess.py`（前処理）、`scripts/verify_variable_length.py --preprocess full --coreml-mel-dir`（合成）
- 第 4 章の計測（`ThesisBenchConfig`、`fixed262`、`trimAndFade`）の設定は変えていない

## range1 × 精度 × Compute Units の実機確認（2026-11-22、iPhone 17 Pro / iOS 27.0）

`scripts/run_variable_length_matrix.py`（1 構成を 1 回の xcodebuild で実行。`VariableLengthMatrixTests`）。入力は代表 4 件
（T = 56 / 262 / 432 / 735）、前処理は正式評価と同じ。比べる基準は PyTorch FP32（同じ前処理）。結果は `metrics/variable_length_matrix_20261122/`。

| 精度 | モデルの形式 |
|---|---|
| Float32 | 重み・活性化とも float32、入出力 float32 |
| Float16 | 重み・活性化とも float16（一部 float32）、入出力 float16 |
| Int8 | 重みだけ int8（`constexpr_blockwise_shift_scale` で読み込み時に float16 に戻す）、活性化 float16（W8A16） |

| 構成 | 結果 | PyTorch FP32 とのログメル差（dB、T=56/262/432/735） | 備考 |
|---|---|---|---|
| F32 × cpuOnly | 4 件とも成功 | 0.00 / 0.01 / 0.08 / 0.09 | メモリ 100 MB 以下。T=735 で Decoder 17.7 秒 |
| F32 × cpuAndNE | 成功。cpuOnly と出力がビット一致 | — | ANE は float32 を扱わない。cpuOnly と同じ構成として扱う |
| F32 × cpuAndGPU | T≤262 は成功、T=432 でプロセスが落ちた | 0.00 / 0.01 / – / – | T=262 の後に 2.1 GB。E5RT の reshape エラーがログに出る（推論は続く）。初回は Decoder が 60 倍遅い（形ごとの特殊化） |
| F32 × all | HiFi-GAN の推論の準備で失敗（全長） | – | `BNNS convolution: Error in shape deduction`。T=735 でメモリ上限（per-process-limit、約 3.4 GB）で強制終了（JetsamEvent で確認） |
| F16 × cpuOnly | 4 件とも成功 | 2.94 / 6.35 / 4.98 / 9.98 | メモリが長さとともに増える（T=735 で 370 MB）。長い入力で PyTorch がクリップする箇所でもクリップしない（振幅が違う） |
| F16 × cpuAndGPU | T≤432 は成功、T=735 で落ちた | 2.03 / 2.32 / 1.38 / – | T=432 の後に 3.2 GB。`Port descriptor cannot be changed` がログに出る |
| F16 × cpuAndNE / all | HiFi-GAN の推論の準備で失敗（全長） | – | cpuAndNE は BNNS の shape deduction、all は Port descriptor のエラー。all は T=735 で落ちた |
| Int8 × cpuOnly | 4 件とも成功 | 10.20 / 6.26 / 11.04 / 14.85 | メモリ 80 MB 以下 |
| Int8 × cpuAndNE | 4 件とも成功 | 10.20 / 6.26 / 11.04 / 14.92 | T≤432 は cpuOnly とビット一致、T=735 だけわずかに違う。MLComputePlan では NE に 190 演算を割り当てる予定だが、実際にどこで動いたかは断定できない |
| Int8 × cpuAndGPU | T≤432 は成功、T=735 で落ちた | 9.09 / 3.80 / 6.16 / – | T=432 の後に 3.3 GB |
| Int8 × all | HiFi-GAN の推論の準備で失敗（全長） | – | T=735 で落ちた |

- 参考: 開発者のオリジナルと PyTorch FP32 の差（計算機の違い）は、T=56 で 0.76 dB
- 以前（2026-05-04、iPhone (3) / iOS 26）の `No memory object bound to port` は再現しなかった。代わりに、GPU では長い入力でメモリが増え続けて落ちる
- GPU の時間はキャッシュの状態で大きく変わる（同じ構成で、初回 10.5 秒、2 回目以降 0.4 秒など）。RTF に使うときは条件をそろえる必要がある
- MLComputePlan の割り当ては予定であり、実際に使われた装置ではない（Int8 × cpuAndNE の例）

## 一括合成経路（正式評価用）

第 4 章の計測（`ThesisBenchConfig`）とは別の経路（`FormalEvalConfig` / `FormalEvalRunner`、起動時の環境変数 `CMLA_FORMAL_EVAL=1`）。

```bash
# 1. 入力を作る（元のデータセットは読むだけ。eval/fad/data/formal_inputs/ に 700 件、manifest.csv 付き）
PronounSE/venv/bin/python scripts/prepare_formal_eval_inputs.py --source ~/Desktop/Vocal_Mimicry
# 2. 端末に送って合成し、取り出して確かめる（アプリはこのブランチのものを入れておく）
PronounSE/venv/bin/python scripts/run_formal_eval.py --device <devicectl の Identifier> --session <ID> \
    --precision Float32 --cu cpuOnly --speakers f-01,m-01,m-02,m-03,m-04,m-05 --push \
    --reference-dir ~/Desktop/Trimmed_Explosion
```

- 出力: `metrics/formal_eval/<session>/device/<話者>/<参照>_<話者>_pronounse_ios.wav`（FAD 用、PCM16）と `..._f32.npy`
  （クリッピングの解析用、PCM16 にする前の float32）、`results.csv`（サンプルごとの時間・統計・失敗）、`launches.csv`、`session.json`
- 時間は前処理（読み込み・メル）、推論（Encoder・Decoder・HiFi-GAN）、後処理、保存を分けて記録する
- 同じ `--session` で起動し直すと、ok のサンプルは飛ばして続きから合成する。条件が違えば何もせずに止まる。既存の出力は上書きしない
- 2026-11-22 の動作確認（F32 × cpuOnly、m-01 の 3 件。2 件 → 起動し直して 3 件目だけ）: 長さ・wav と npy の対応・参照音の名前の対応に問題なし。
  1 件を同じ条件の PyTorch と比べて SNR 83.0 dB

## GPU・ANE の切り分け（2026-11-22、iPhone 17 Pro / iOS 27.0）

`VariableLengthDiagnosticsTests`（モデルごとの Compute Units、Decoder 入力の作り方、reshapeFrequency、autoreleasepool、形の掃引）を
`scripts/run_variable_length_diagnostics.py` で 1 構成ずつ実行。結果は `metrics/gpu_ane_diag_20261122/`。

### 確認した事実

1. **FP32 の Encoder・Decoder は、CPU で T = 65〜80 のとき BNNS の中で落ちる（SIGSEGV）**。
   - Encoder は T=65〜80、Decoder は T=66〜80（74 を除く。境界は実行ごとに揺れる）。T=48〜64・81〜735 は落ちない
   - 最初の呼び出しでも落ちる（形だけで決まる。値には依らない）。Mac の CPU では再現しない
   - 正式評価の入力 700 件のうち 21 件（評価 6 話者 18 件、m-00 3 件）が該当し、一括合成経路の F32 × cpuOnly でも実際に落ちた
   - cpuAndNE（FP32 は CPU で動く）も同じ。cpuAndGPU と all（GPU で動く）は落ちない。reshapeFrequency では回避できない
   - FP16・Int8 の CPU は T=48〜735 で落ちない
2. **GPU のメモリ増加は、Decoder の入力の形が毎ステップ変わることによる**。
   - Decoder だけ GPU（入力を 1 フレームずつ伸ばす）で 100 ステップあたり約 160 MB 増え、T=262 の途中で 1.5 GB を超える
   - autoreleasepool・reshapeFrequency=frequent では変わらない。モデルを解放すると大部分が戻る（形ごとの GPU の資源をモデルが持っていると推測）
   - Decoder 入力を毎回 T フレームにする（padT。因果マスクにより結果は同じ。Mac で差は 1e-4 以下）と、1 サンプルの中でメモリは増えない
   - 異なる T を 60 通り続けて流すと、GPU を使う構成では形ごとにメモリが増える（FP16 で 1 形あたり約 9〜19 MB）。モデルを解放すると戻る
3. **ANE は range1 では実際には使われていない**（使われた証拠が無い）。
   - Encoder・Decoder・HiFi-GAN を cpuAndNE にしても、出力が cpuOnly とビット一致し、時間も同じ（FP16・Int8 の B、Int8 の H）。
     MLComputePlan は NE に 97〜190 演算を割り当てる予定と出すが、実行はされていない
   - FP16 の HiFi-GAN は cpuAndNE / all で推論の準備に失敗する（`BNNS convolution: Error in shape deduction`、all は
     `Port descriptor cannot be changed`）。既存の `range16`（下限 16）でも同じ
   - 以前に Int8 × cpuAndNE の T=735 だけ cpuOnly と違ったのは、Int8 × cpuOnly 自体が T=735 で実行ごとに結果が変わるため（ANE の証拠ではない）

### 一括合成経路での確認（m-01・m-03 の 30 件 + T=65〜80 の 6 件）

実験用の設定（`CMLA_FORMAL_CU_ENC/_DEC/_HIFI`、`CMLA_FORMAL_DEC_INPUT=padT`、`CMLA_FORMAL_RELOAD_MB`。指定しなければ従来の動作）で実行。

| 構成 | 結果 | メモリ最大 | 推論の平均（T 平均 441） | 数値 |
|---|---|---|---|---|
| F32: Encoder GPU・Decoder GPU（padT）・HiFi-GAN CPU | 36 件すべて成功（T=65〜80 も） | 251 MB | 24.1 秒（Decoder 23.7 秒） | F32 × cpuOnly と SNR 76〜83 dB |
| F16: Encoder GPU・Decoder CPU・HiFi-GAN GPU | 36 件すべて成功 | 203 MB | 4.7 秒 | — |
| F16: Decoder GPU（padT）・他 CPU | 36 件すべて成功 | 427 MB | 5.5 秒 | — |

- 30 件では読み込み直し（1200 MB）は起きなかった。600 件（T は 352 通り）での増え方は未確認

## 正式評価 第 1 段階の確認（2026-11-23）

### 採用する構成（暫定）

| | 精度（重み/活性化） | Encoder | Decoder | HiFi-GAN | Decoder 入力 |
|---|---|---|---|---|---|
| A | FP32 / FP32 | cpuAndGPU | cpuAndGPU | cpuOnly | padT（T フレームに 0 埋め） |
| B | FP16 / FP16 | cpuAndGPU | cpuOnly | cpuAndGPU | grow |
| C | INT8 / FP16（W8A16） | cpuOnly | cpuOnly | cpuOnly | grow |

### PyTorch FP32 との比較（一括合成の出力、`scripts/compare_formal_to_pytorch.py`）

| 構成 | 件数 | ログメル差 中央 / 最大 | 波形 SNR 中央 / 最小 |
|---|---|---|---|
| A（FP32 GPU、padT） | 36（T=65〜80 の 6 件を含む） | 0.00 / 0.17 dB | 60〜66 / 37.9 dB |
| B（FP16） | 36 | 3.7〜4.5 / 16.7 dB | 約 0 dB |
| 参考: FP16 Decoder GPU（padT） | 36 | 2.5〜3.7 / 15.7 dB | 約 0 dB |
| C（INT8 cpuOnly） | 6 | 3.7 / 8.1 dB | 約 −1 dB |

### A の連続実行（診断用、`metrics/formal_eval_diag/`）

- 100 件（6 話者、T は 94 通り、48〜735、T=65〜80 の 18 件を含む）がすべて成功。NaN・Inf 0
- メモリ 88〜338 MB（最初の 10 件の平均 182 MB → 最後の 10 件 312 MB。ゆるやかに増える）。1500 MB での読み込み直しは起きなかった
- 21 件目の途中でアプリを強制終了 → 起動し直すと残りの 79 件を合成。重複・書きかけのファイルの残りなし
- 毎回モデルを読み込み直した 10 件は、連続実行の出力とビット一致（実行ごとの再現性もある）
- 記録の無い出力が既にあるサンプルは skipped_existing_output として飛ばし、既存のファイルは変わらなかった
- C（INT8 cpuOnly）の T=725〜735 の 3 件を 3 回ずつ合成し、すべてビット一致（以前の不一致は、別々のテストの経路どうしの比較だった）

### 開発者の FAD との比較可能性（`reproduce_original_fad.py`、`results/original_fad_repro_20261123/`）

提供されたオリジナルの合成音で本実装の FAD を計算した（Wavegram-Logmel-CNN14、長さは切らない、float64）。

| 条件 | 22050 Hz 経由（本実装） | 32 kHz へ直接（kadtk の流れに近い） | 開発者 |
|---|---:|---:|---:|
| m-01 | 17.16 | 17.59 | 17.85 |
| m-02 | 16.57 | 16.97 | 17.05 |
| m-03 | 17.40 | 17.95 | 17.28 |
| m-04 | 17.11 | 17.31 | 16.65 |
| m-05 | 19.22 | 19.17 | 19.29 |
| f-01 | 19.62 | 19.57 | 19.28 |
| Whole | 16.29 | 16.59 | 14.09 |

- 話者別は開発者の値と ±0.7 以内（平均の差 0.35〜0.36）。完全には一致しない
- Whole だけ 2.2〜2.5 大きい。参照音を 6 回繰り返す（16.23）・共分散の分母 n（16.21）・話者別の平均（17.85）でも 14.09 にならない
- CNN14（32k）での確認は、重みの取得（Zenodo）が途中で切れたため未実施
- したがって、iOS 版の FAD は本実装で計算したオリジナルの FAD（上の表の「22050 Hz 経由」）と比べる。
  開発者の値は参考として並べ、Whole の差は計算条件の違いの可能性があることを明記する

### FAD の暫定基準（2026-11-23 決定）

今後の主な比較では、同じ計算実装（本 README の方法: Wavegram-Logmel-CNN14、22050 Hz 経由で 32 kHz、長さは切らない、
float64・不偏共分散）で計算したオリジナル PronounSE の FAD を暫定の基準にする。

| 条件 | 本実装で計算したオリジナルの FAD（暫定基準） | 開発者が報告した値（参考） |
|---|---:|---:|
| m-01 | 17.16 | 17.85 |
| m-02 | 16.57 | 17.05 |
| m-03 | 17.40 | 17.28 |
| m-04 | 17.11 | 16.65 |
| m-05 | 19.22 | 19.29 |
| f-01 | 19.62 | 19.28 |
| Whole | 16.29 | 14.09 |

**未解決の点**: 開発者の計算に使われた PANNs のモデル（本実装と同じ Wavegram-Logmel-CNN14 か）、
標本化周波数の変換と長さの扱い、Whole の作り方（本実装は参照音 100 件 vs 6 話者の合成音 600 件）が確認できていない。
話者別は ±0.7 以内で近いが一致はせず、Whole は 2.2 の差がある。開発者に確認するまで、開発者の値は参考として扱う。

## 正式評価 第 2 段階：FP32 の 600 件合成（2026-11-23、iPhone 17 Pro / iOS 27.0）

- 出力: `metrics/formal_eval_official/fp32_A_20261123/`（`device/<話者>/` に WAV（PCM16）と NPY（float32）、
  `device/results.csv`・`memory.csv`・`session.json`、ホスト側の `run_config.json`（設定・入力一覧・モデルの SHA-256）、
  `verify.json`・`verify_full.json`）
- 条件: range1・FP32・fullInput（`formal_inputs` の 22050 Hz float32 WAV）。Encoder cpuAndGPU、Decoder cpuAndGPU（padT）、
  HiFi-GAN cpuOnly。Decoder は T 回、出力は T×256 サンプル、デエンファシスの後に PCM16。1500 MB で読み込み直す設定
  （FP32×cpuOnly ではない）
- 結果: 600/600 成功、失敗・再試行 0。話者ごとに 100 件。WAV と NPY は一対一、22050 Hz・モノラル・PCM16、
  長さは全件 T×256（T は 48〜735、中央 288）、NaN・Inf 0、欠損・重複 0、元の口真似音は台帳の SHA-256 と一致
- 振幅: ピークの中央 0.954、最大 2.199。|x|≥1 は 256 ファイル・43,294 サンプル（PyTorch 版も同様に超えるので不具合とはしない）
- メモリ: 100〜331 MB。入力の長さの最大値が更新されたとき（T=727、735）だけ段状に増え、その後は横ばい。読み込み直しは 0 回
- 時間（中央 / 最大）: 前処理 67 / 188 ms、Encoder 16 / 34 ms、Decoder 11.8 / 76.0 s、HiFi-GAN 155 / 482 ms、
  後処理 5 / 17 ms。Decoder 1 ステップあたりの中央値は 100 件ごとに 37〜42 ms で、実行中に遅くなる傾向は無い
- PyTorch FP32 との比較（キャッシュのある 44 件）: ログメル差 中央 0.003 / 最大 0.17 dB、SNR 中央 62.7 / 最小 31.1 dB
- FAD はまだ計算していない。暫定の基準は上の「FAD の暫定基準」

## 正式評価 第 3 段階：FP16 の 600 件合成（2026-11-23、iPhone 17 Pro / iOS 27.0）

- 出力: `metrics/formal_eval_official/fp16_B_20261123/`（構成は FP32 と同じ。`run_config.json` に FP16 モデル 3 つの SHA-256）
- 条件: range1・FP16・fullInput。Encoder cpuAndGPU、Decoder cpuOnly（0 埋めしない可変長入力 grow）、HiFi-GAN cpuAndGPU。
  1500 MB で読み込み直す設定
- 結果: 600/600 成功、失敗・再試行 0。話者ごとに 100 件。入力の集合と T は FP32 と全件同じ。
  WAV と NPY は一対一、22050 Hz・モノラル・PCM16、長さは全件 T×256、NaN・Inf 0、欠損・重複 0、元の口真似音は変わっていない
- 振幅: ピークの中央 0.948（FP32 は 0.954）、最大 2.385。|x|≥1 は 250 ファイル・35,273 サンプル（FP32 は 256 ファイル・43,294 サンプル）。
  ピークの比（FP16 / FP32）は中央 0.995、5〜95% で 0.76〜1.35。波形の違いだけでは音質の良し悪しは判断しない（FAD で評価する）
- メモリ: 109〜584 MB。読み込み直しは 0 回だが、100 件あたり約 71 MB ずつ増え続けた（FP32 の構成 A は横ばい）。
  新しい長さの入力が来るたびに増えるので、GPU で動かす Encoder・HiFi-GAN の形ごとの確保と考えられる（未確認）
- 時間（中央 / 平均 / 最大）: Decoder 1.96 / 2.44 / 13.6 s、HiFi-GAN 156 / 156 / 386 ms、Encoder 9 / 9 / 156 ms、
  前処理 79 ms、後処理 7 ms、推論全体 2.22 / 2.72 / 14.3 s（FP32 の約 1/5）
- 実行中の減速: 同じ参照音（T≈350）で Decoder の 1 ステップが 5.0 ms（最初）から 9.6 ms（最後）に遅くなった。
  最後の 10 件を新しいプロセスで合成し直す（`metrics/formal_eval/diag_fp16_B_fresh_20261123/`、診断用）と、
  出力はビット一致、メモリは約 100 MB に戻り、時間は約 25% 短くなったが、最初の速さには戻らなかった。
  プロセスの状態（メモリの増加）と端末の発熱の両方が効いていると考えられる。出力には影響しないが、
  この 600 件の時間は実行順に左右されるので、RTF には使わない
- FAD はまだ計算していない

## 正式評価 第 4 段階：INT8 の 600 件合成（2026-11-23、iPhone 17 Pro / iOS 27.0）

- 出力: `metrics/formal_eval_official/int8_C_20261123/`（構成は FP32・FP16 と同じ）
- 条件: range1・INT8（W8A16）・fullInput。Encoder・Decoder・HiFi-GAN すべて cpuOnly、Decoder 入力 grow。1500 MB で読み込み直す設定
- 結果: 600/600 成功、失敗・再試行 0。話者ごとに 100 件。入力ファイル・入力のハッシュ・T は FP32・FP16 と全件同じ。
  WAV と NPY は一対一（WAV は NPY を ±1 で切った PCM16 と 1 LSB 以内で一致）、22050 Hz・モノラル・PCM16、長さは全件 T×256、
  NaN・Inf 0、欠損・重複・書きかけ 0、元の口真似音は変わっていない
- 振幅: ピークの中央 0.956、最大 2.084。|x|≥1 は 264 ファイル・42,020 サンプル（FP32 256・43,294、FP16 250・35,273）。
  ピークの比の中央は INT8/FP32 0.997、INT8/FP16 1.007
- メモリ: 14.5〜43.2 MB（phys_footprint）、読み込み直し 0 回。100 件ごとの中央は 19.8 → 35.4 MB と少しずつ増えた（100 件あたり約 3 MB）
- 時間（中央 / 平均 / 最大）: Decoder 2.05 / 2.56 / 14.1 s、HiFi-GAN 103 / 107 / 324 ms、Encoder 6 / 6 / 19 ms、
  推論全体 2.26 / 2.80 / 15.0 s。同じ入力どうしの比の中央は FP32/INT8 5.04、FP16/INT8 0.98
- 実行中の減速: Decoder の 1 ステップは 100 件ごとの中央で 4.95 → 9.07 ms。メモリがほとんど増えていないのに FP16 と同じくらい遅くなったので、
  FP16 の減速もメモリだけでは説明できず、端末の発熱が主な原因の可能性が高い（温度は記録していないので未確認）。
  この 600 件の時間は RTF には使わない
- FAD はまだ計算していない
