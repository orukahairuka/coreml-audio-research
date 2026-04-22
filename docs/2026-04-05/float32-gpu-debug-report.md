# Float32 + GPU/ANE 実行時の音声異常 調査レポート

## 現象

Float32 モデルを CPU+GPU または CPU+GPU+ANE で実行すると、以下が発生する：

- 合成音声が異常（変な音になる）
- 振幅グラフの値が異常に大きくなる

Float16 / Int8 では全 compute unit で正常動作する。

## デバッグ統計

### Float32, CPU+GPU（異常）

```
[Encoder output]
min=-5.1483e+00  max=9.7835e+00  mean=3.4685e-02

[Decoder steps] (15 recorded)
Step 0
  mel_out:     min=-2.0455e-03  max=1.1303e-03  mean=-5.6139e-05
  postnet_out: min=-2.0773e-03  max=1.1012e-03  mean=-5.4456e-05
Step 1
  mel_out:     min=-2.0451e-03  max=5.4048e-01  mean=1.6723e-01
  postnet_out: min=-2.0769e-03  max=5.4031e-01  mean=1.6721e-01
Step 2
  mel_out:     min=-2.0451e-03  max=6.5737e-01  mean=2.8504e-01
  postnet_out: min=-2.0770e-03  max=6.5725e-01  mean=2.8503e-01
Step 3
  mel_out:     min=-2.0453e-03  max=6.6544e-01  mean=3.3989e-01
  postnet_out: min=-2.0771e-03  max=6.6537e-01  mean=3.3986e-01
Step 4
  mel_out:     min=-2.0451e-03  max=6.8542e-01  mean=3.6858e-01
  postnet_out: min=-2.0769e-03  max=6.8541e-01  mean=3.6856e-01
Step 129
  mel_out:     min=-2.5609e-03  max=7.3167e-01  mean=4.1118e-01
  postnet_out: min=-2.5757e-03  max=7.3145e-01  mean=4.1117e-01
Step 130
  mel_out:     min=-2.5606e-03  max=7.3167e-01  mean=4.1055e-01
  postnet_out: min=-2.5754e-03  max=7.3145e-01  mean=4.1055e-01
Step 131
  mel_out:     min=-2.5606e-03  max=7.3167e-01  mean=4.0980e-01
  postnet_out: min=-2.5754e-03  max=7.3145e-01  mean=4.0979e-01
Step 132
  mel_out:     min=-2.5614e-03  max=7.3167e-01  mean=4.0909e-01
  postnet_out: min=-2.5761e-03  max=7.3145e-01  mean=4.0908e-01
Step 133
  mel_out:     min=-2.5605e-03  max=7.3167e-01  mean=4.0836e-01
  postnet_out: min=-2.5752e-03  max=7.3145e-01  mean=4.0836e-01
Step 257
  mel_out:     min=-5.3261e-02  max=7.3167e-01  mean=2.7386e-01
  postnet_out: min=-5.3346e-02  max=7.3145e-01  mean=2.7385e-01
Step 258
  mel_out:     min=-5.3261e-02  max=7.3167e-01  mean=2.7280e-01
  postnet_out: min=-5.3346e-02  max=7.3145e-01  mean=2.7280e-01
Step 259
  mel_out:     min=-5.3261e-02  max=7.3167e-01  mean=2.7175e-01
  postnet_out: min=-5.3346e-02  max=7.3145e-01  mean=2.7175e-01
Step 260
  mel_out:     min=-5.3261e-02  max=7.3167e-01  mean=2.7071e-01
  postnet_out: min=-5.3346e-02  max=7.3145e-01  mean=2.7071e-01
Step 261
  mel_out:     min=-5.3261e-02  max=7.3167e-01  mean=2.6967e-01
  postnet_out: min=-5.3346e-02  max=7.3145e-01  mean=2.6967e-01

[HiFi-GAN input]
min=-5.3346e-02  max=7.3145e-01  mean=2.6967e-01
[HiFi-GAN output]
min=-3.7808e-02  max=9.9999e-01  mean=4.7361e-01

[Waveform (before de-emphasis)]
min=-3.7808e-02  max=9.9999e-01  mean=4.7361e-01
[Waveform (after de-emphasis)]
min=6.4154e-01  max=3.3197e+01  mean=1.5787e+01
```

### Float32, CPU only（正常）

```
[Encoder output]
min=-5.1483e+00  max=9.7835e+00  mean=3.4685e-02

[Decoder steps] (15 recorded)
Step 0
  mel_out:     min=-2.0452e-03  max=1.1299e-03  mean=-5.6135e-05
  postnet_out: min=-2.0771e-03  max=1.1008e-03  mean=-5.4452e-05
Step 1
  mel_out:     min=-2.0454e-03  max=5.4048e-01  mean=1.6723e-01
  postnet_out: min=-2.0773e-03  max=5.4031e-01  mean=1.6721e-01
Step 2
  mel_out:     min=-2.0454e-03  max=6.5737e-01  mean=2.8504e-01
  postnet_out: min=-2.0773e-03  max=6.5725e-01  mean=2.8503e-01
Step 3
  mel_out:     min=-2.0452e-03  max=6.6544e-01  mean=3.3989e-01
  postnet_out: min=-2.0771e-03  max=6.6537e-01  mean=3.3986e-01
Step 4
  mel_out:     min=-2.0452e-03  max=6.8542e-01  mean=3.6858e-01
  postnet_out: min=-2.0771e-03  max=6.8541e-01  mean=3.6856e-01
Step 129
  mel_out:     min=-2.5607e-03  max=7.3167e-01  mean=4.1118e-01
  postnet_out: min=-2.5755e-03  max=7.3145e-01  mean=4.1117e-01
Step 130
  mel_out:     min=-2.5606e-03  max=7.3167e-01  mean=4.1055e-01
  postnet_out: min=-2.5754e-03  max=7.3145e-01  mean=4.1055e-01
Step 131
  mel_out:     min=-2.5606e-03  max=7.3167e-01  mean=4.0980e-01
  postnet_out: min=-2.5754e-03  max=7.3145e-01  mean=4.0979e-01
Step 132
  mel_out:     min=-2.5606e-03  max=7.3167e-01  mean=4.0909e-01
  postnet_out: min=-2.5753e-03  max=7.3145e-01  mean=4.0908e-01
Step 133
  mel_out:     min=-2.5607e-03  max=7.3167e-01  mean=4.0836e-01
  postnet_out: min=-2.5755e-03  max=7.3145e-01  mean=4.0836e-01
Step 257
  mel_out:     min=-5.3261e-02  max=7.3167e-01  mean=2.7386e-01
  postnet_out: min=-5.3347e-02  max=7.3145e-01  mean=2.7385e-01
Step 258
  mel_out:     min=-5.3261e-02  max=7.3167e-01  mean=2.7280e-01
  postnet_out: min=-5.3347e-02  max=7.3145e-01  mean=2.7280e-01
Step 259
  mel_out:     min=-5.3261e-02  max=7.3167e-01  mean=2.7175e-01
  postnet_out: min=-5.3347e-02  max=7.3145e-01  mean=2.7175e-01
Step 260
  mel_out:     min=-5.3261e-02  max=7.3167e-01  mean=2.7071e-01
  postnet_out: min=-5.3347e-02  max=7.3145e-01  mean=2.7071e-01
Step 261
  mel_out:     min=-5.3261e-02  max=7.3167e-01  mean=2.6967e-01
  postnet_out: min=-5.3346e-02  max=7.3145e-01  mean=2.6967e-01

[HiFi-GAN input]
min=-5.3346e-02  max=7.3145e-01  mean=2.6967e-01
[HiFi-GAN output]
min=-4.1145e-01  max=3.7023e-01  mean=-9.9091e-05

[Waveform (before de-emphasis)]
min=-4.1145e-01  max=3.7023e-01  mean=-9.9091e-05
[Waveform (after de-emphasis)]
min=-7.4241e-01  max=7.4181e-01  mean=-3.3031e-03
```

## 分析

### Encoder / Decoder は正常（CPU+GPU と CPU only で一致）

- Encoder 出力: min=-5.15, max=9.78 — 正常な範囲
- Decoder 各ステップ: 値は [-0.05, 0.73] の範囲で安定。NaN/Inf なし
- 両条件で有効数字4桁まで一致 → Encoder/Decoder は GPU 上でも正しく動作

### HiFi-GAN の出力が GPU 実行時のみ異常

HiFi-GAN の Generator は最終層に `tanh` を使用しており、出力は [-1, 1] の範囲で 0 を中心に振動する波形になるべきである。

|  | CPU only（正常） | CPU+GPU（異常） |
|---|---|---|
| HiFi-GAN min | **-0.411** | -0.038 |
| HiFi-GAN max | **0.370** | 1.000 |
| HiFi-GAN mean | **-0.0001** | 0.474 |

CPU+GPU では HiFi-GAN 出力がほぼ [0, 1] に偏っている。`tanh` の出力 [-1, 1] が [0, 1] にシフトしたような挙動。

### デエンファシスフィルタの暴走

デエンファシスフィルタ `y[n] = x[n] + 0.97 * y[n-1]` は、入力が 0 中心で振動することを前提としている。入力がほぼ正の値のみの場合、累積加算により値が際限なく増大する：

|  | CPU only（正常） | CPU+GPU（異常） |
|---|---|---|
| 最終波形 min | -0.742 | 0.642 |
| 最終波形 max | 0.742 | **33.2** |
| 最終波形 mean | -0.003 | **15.8** |

これが振幅グラフの値がおかしくなる直接原因である。

## 結論

### これは CoreML ランタイムのバグ

根拠：

1. **同じモデル、同じ入力**で CPU only なら正常、CPU+GPU で異常 → コード側の問題ではない
2. **同じモデル構造**で Float16 なら全 compute unit で正常 → モデル構造の問題ではない
3. **Float32 + GPU の組み合わせだけ** HiFi-GAN の出力がおかしい → CoreML が Float32 の特定オペレーション（`tanh` か `leaky_relu` か `ConvTranspose1d` か）を GPU にディスパッチする際に不正な結果を返している

HiFi-GAN は `ConvTranspose1d`（アップサンプリング）、`leaky_relu`、`tanh` を多用する複雑なモデルで、Transformer の Encoder/Decoder よりオペレーションの種類が多い。GPU 上の Float32 実行パスでこれらのいずれかにバグがあると考えられる。

### Float16 / Int8 が GPU でも正常な理由

Apple Silicon の GPU と ANE は Float16 に最適化されて設計されている。

| 精度 | GPU での実行 | 理由 |
|---|---|---|
| **Float16** | ネイティブ実行 → 問題なし | GPU/ANE の最も基本的な演算パスで、最もテストされている |
| **Int8** | 重みは Int8、演算時は Float16 にデクオンタイズ → 問題なし | 実際の計算パスは Float16 と同じ |
| **Float32** | GPU が Float32 を処理する別のコードパス → **バグあり** | Metal は Float32 対応だが、CoreML ランタイムの Float32→GPU ディスパッチに問題がある |

CoreML の mlprogram 形式は `compute_precision` のデフォルトが Float16 で、Apple 自身も Float16 での GPU/ANE 実行を推奨している。Float32 + GPU は「できるが推奨されない」組み合わせであり、テスト・最適化が不十分なコードパスにバグが残っていると考えられる。
