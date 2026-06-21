HiFi-GAN 入力（postnet）: `audio/iPhone_3_phase1_20260519/debug/20260519_212806_f16GpuRepeat2_Float16_cpuAndGPU/postnet_output.npy`  shape=(1, 256, 262)

基準 = PyTorch HiFi-GAN（同じ postnet 入力）。判定は波形なので振幅(rms_ratio)主。

| stage | condition | MAE | RMSE | cosine | min | max | mean | std | peak | rms | NaN/Inf | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| HiFi-GAN出力 | PyTorch（基準） | 0.0000e+00 | 0.0000e+00 | 1.0000 | -0.4584 | 0.3346 | -9.876e-05 | 0.02689 | 0.4584 | 0.02689 | なし | 🟢 正常 |
| HiFi-GAN出力 | CoreML F32 cpuOnly | 3.6517e-07 | 9.4313e-07 | 1.0000 | -0.4584 | 0.3346 | -9.876e-05 | 0.02689 | 0.4584 | 0.02689 | なし | 🟢 正常 |
| HiFi-GAN出力 | CoreML F16 cpuAndGPU | 9.5800e-05 | 2.7213e-04 | 1.0000 | -0.4575 | 0.3418 | -9.983e-05 | 0.02683 | 0.4575 | 0.02683 | なし | 🟢 正常 |
| HiFi-GAN出力 | CoreML F16 cpuAndNE | 4.9361e-02 | 1.2010e-01 | 0.0067 | -0.9692 | 0.9683 | -0.0004029 | 0.1172 | 0.9692 | 0.1172 | なし | 🔴 破綻 |
| HiFi-GAN出力 | CoreML Int8 cpuAndGPU | 6.6170e-04 | 1.7314e-03 | 0.9979 | -0.4641 | 0.364 | -9.56e-05 | 0.02682 | 0.4641 | 0.02682 | なし | 🟢 正常 |
| HiFi-GAN出力 | CoreML Int8 cpuAndNE | 4.9473e-02 | 1.2035e-01 | 0.0067 | -0.9712 | 0.9702 | -0.0003685 | 0.1175 | 0.9712 | 0.1175 | なし | 🔴 破綻 |

### 実機 ANE 真値との照合

- PyTorch HiFi-GAN 出力 rms = 0.026885
- 実機 ANE 壊れ run の HiFi-GAN 出力 rms = 0.11403 （`20260519_215227_f16NeRepeat1_Float16_cpuAndNE`）→ PyTorch 比 4.24×
- Mac ローカルの cpuAndNE が上の実機 rms 跳ねを再現できていなければ、「Mac の ANE ≠ iPhone の ANE」であり、真値は実機 npy 側に置く。

