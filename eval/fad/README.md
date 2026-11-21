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
