# Phase 4 — MLComputePlan dispatch マップ集計

集計対象: 18 plan files

生成元スクリプト: `scripts/aggregate_compute_plan.py`

## 1. ディスパッチサマリー（model × precision × computeUnits）

| model | precision | computeUnits | ops | cpu | gpu | NE | unknown |
|---|---|---|---:|---:|---:|---:|---:|
| HiFiGAN_Generator_float16_fixed262 | Float16 | all | 767 | 11 | 124 | 61 | 571 |
| HiFiGAN_Generator_float16_fixed262 | Float16 | cpuAndGPU | 767 | 8 | 188 | 0 | 571 |
| HiFiGAN_Generator_float16_fixed262 | Float16 | cpuAndNE | 767 | 115 | 0 | 81 | 571 |
| HiFiGAN_Generator_float32_fixed262 | Float32 | cpuAndGPU | 767 | 7 | 189 | 0 | 571 |
| HiFiGAN_Generator_int8_fixed262 | Int8 | all | 767 | 11 | 124 | 61 | 571 |
| HiFiGAN_Generator_int8_fixed262 | Int8 | cpuAndNE | 767 | 115 | 0 | 81 | 571 |
| Transformer_Decoder_float16_fixed262 | Float16 | all | 516 | 16 | 0 | 188 | 312 |
| Transformer_Decoder_float16_fixed262 | Float16 | cpuAndGPU | 516 | 0 | 204 | 0 | 312 |
| Transformer_Decoder_float16_fixed262 | Float16 | cpuAndNE | 516 | 16 | 0 | 188 | 312 |
| Transformer_Decoder_float32_fixed262 | Float32 | all | 498 | 0 | 195 | 0 | 303 |
| Transformer_Decoder_float32_fixed262 | Float32 | cpuAndGPU | 498 | 0 | 195 | 0 | 303 |
| Transformer_Decoder_float32_fixed262 | Float32 | cpuAndNE | 498 | 195 | 0 | 0 | 303 |
| Transformer_Decoder_float32_fixed262 | Float32 | cpuOnly | 498 | 195 | 0 | 0 | 303 |
| Transformer_Decoder_int8_fixed262 | Int8 | all | 523 | 20 | 0 | 188 | 315 |
| Transformer_Decoder_int8_fixed262 | Int8 | cpuAndGPU | 523 | 0 | 208 | 0 | 315 |
| Transformer_Decoder_int8_fixed262 | Int8 | cpuAndNE | 523 | 20 | 0 | 188 | 315 |
| Transformer_Encoder_float16_fixed262 | Float16 | cpuAndNE | 258 | 7 | 0 | 97 | 154 |
| Transformer_Encoder_float32_fixed262 | Float32 | cpuAndGPU | 252 | 0 | 101 | 0 | 151 |

## 3. 各 plan の GPU 行き最初の op（Phase 3 観測候補）

| model | precision | computeUnits | first GPU op index | operatorName | outputName | weight |
|---|---|---|---:|---|---|---:|
| HiFiGAN_Generator_float16_fixed262 | Float16 | all | 226 | ios18.leaky_relu | input_87_cast_fp16 | 0.0014 |
| HiFiGAN_Generator_float16_fixed262 | Float16 | cpuAndGPU | 28 | ios18.conv | input_9_cast_fp16 | 0.0044 |
| HiFiGAN_Generator_float16_fixed262 | Float16 | cpuAndNE | - | (none) | - | - |
| HiFiGAN_Generator_float32_fixed262 | Float32 | cpuAndGPU | 178 | ios18.conv | input_9 | 0.0037 |
| HiFiGAN_Generator_int8_fixed262 | Int8 | all | 226 | ios18.leaky_relu | input_87_cast_fp16 | 0.0014 |
| HiFiGAN_Generator_int8_fixed262 | Int8 | cpuAndNE | - | (none) | - | - |
| Transformer_Decoder_float16_fixed262 | Float16 | all | - | (none) | - | - |
| Transformer_Decoder_float16_fixed262 | Float16 | cpuAndGPU | 3 | ios18.linear | linear_0_cast_fp16 | 0.0062 |
| Transformer_Decoder_float16_fixed262 | Float16 | cpuAndNE | - | (none) | - | - |
| Transformer_Decoder_float32_fixed262 | Float32 | all | 98 | ios18.linear | input_1 | 0.0056 |
| Transformer_Decoder_float32_fixed262 | Float32 | cpuAndGPU | 98 | ios18.linear | input_1 | 0.0056 |
| Transformer_Decoder_float32_fixed262 | Float32 | cpuAndNE | - | (none) | - | - |
| Transformer_Decoder_float32_fixed262 | Float32 | cpuOnly | - | (none) | - | - |
| Transformer_Decoder_int8_fixed262 | Int8 | all | - | (none) | - | - |
| Transformer_Decoder_int8_fixed262 | Int8 | cpuAndGPU | 3 | ios18.linear | linear_0_cast_fp16 | 0.0057 |
| Transformer_Decoder_int8_fixed262 | Int8 | cpuAndNE | - | (none) | - | - |
| Transformer_Encoder_float16_fixed262 | Float16 | cpuAndNE | - | (none) | - | - |
| Transformer_Encoder_float32_fixed262 | Float32 | cpuAndGPU | 49 | ios18.linear | input_1 | 0.0102 |

## 4. F32 Decoder の cpuOnly vs cpuAndGPU で配置が変わる op

（F32 Decoder の cpuOnly と cpuAndGPU が両方そろっていない）

## 2. operator 種別 × device の分布

各 (model, precision, computeUnits) について、operator 種別ごとに
どのデバイスに行ったか。GPU 行きの最初の op を Phase 3 で重点的に見る。

### HiFiGAN_Generator_float16_fixed262 / Float16 / all

| operator | cpu | gpu | NE | unknown | total |
|---|---:|---:|---:|---:|---:|
| const | 0 | 0 | 0 | 571 | 571 |
| ios18.conv | 4 | 53 | 17 | 0 | 74 |
| ios18.leaky_relu | 4 | 47 | 18 | 0 | 69 |
| ios18.add | 0 | 20 | 24 | 0 | 44 |
| ios18.conv_transpose | 2 | 1 | 1 | 0 | 4 |
| ios18.mul | 1 | 2 | 1 | 0 | 4 |
| ios18.tanh | 0 | 1 | 0 | 0 | 1 |

### HiFiGAN_Generator_float16_fixed262 / Float16 / cpuAndGPU

| operator | cpu | gpu | NE | unknown | total |
|---|---:|---:|---:|---:|---:|
| const | 0 | 0 | 0 | 571 | 571 |
| ios18.conv | 1 | 73 | 0 | 0 | 74 |
| ios18.leaky_relu | 4 | 65 | 0 | 0 | 69 |
| ios18.add | 0 | 44 | 0 | 0 | 44 |
| ios18.conv_transpose | 3 | 1 | 0 | 0 | 4 |
| ios18.mul | 0 | 4 | 0 | 0 | 4 |
| ios18.tanh | 0 | 1 | 0 | 0 | 1 |

### HiFiGAN_Generator_float16_fixed262 / Float16 / cpuAndNE

| operator | cpu | gpu | NE | unknown | total |
|---|---:|---:|---:|---:|---:|
| const | 0 | 0 | 0 | 571 | 571 |
| ios18.conv | 57 | 0 | 17 | 0 | 74 |
| ios18.leaky_relu | 51 | 0 | 18 | 0 | 69 |
| ios18.add | 0 | 0 | 44 | 0 | 44 |
| ios18.conv_transpose | 3 | 0 | 1 | 0 | 4 |
| ios18.mul | 3 | 0 | 1 | 0 | 4 |
| ios18.tanh | 1 | 0 | 0 | 0 | 1 |

### HiFiGAN_Generator_float32_fixed262 / Float32 / cpuAndGPU

| operator | cpu | gpu | NE | unknown | total |
|---|---:|---:|---:|---:|---:|
| const | 0 | 0 | 0 | 571 | 571 |
| ios18.conv | 1 | 73 | 0 | 0 | 74 |
| ios18.leaky_relu | 3 | 66 | 0 | 0 | 69 |
| ios18.add | 0 | 44 | 0 | 0 | 44 |
| ios18.conv_transpose | 3 | 1 | 0 | 0 | 4 |
| ios18.mul | 0 | 4 | 0 | 0 | 4 |
| ios18.tanh | 0 | 1 | 0 | 0 | 1 |

### HiFiGAN_Generator_int8_fixed262 / Int8 / all

| operator | cpu | gpu | NE | unknown | total |
|---|---:|---:|---:|---:|---:|
| const | 0 | 0 | 0 | 494 | 494 |
| ios18.constexpr_blockwise_shift_scale | 0 | 0 | 0 | 77 | 77 |
| ios18.conv | 4 | 53 | 17 | 0 | 74 |
| ios18.leaky_relu | 4 | 47 | 18 | 0 | 69 |
| ios18.add | 0 | 20 | 24 | 0 | 44 |
| ios18.conv_transpose | 2 | 1 | 1 | 0 | 4 |
| ios18.mul | 1 | 2 | 1 | 0 | 4 |
| ios18.tanh | 0 | 1 | 0 | 0 | 1 |

### HiFiGAN_Generator_int8_fixed262 / Int8 / cpuAndNE

| operator | cpu | gpu | NE | unknown | total |
|---|---:|---:|---:|---:|---:|
| const | 0 | 0 | 0 | 494 | 494 |
| ios18.constexpr_blockwise_shift_scale | 0 | 0 | 0 | 77 | 77 |
| ios18.conv | 57 | 0 | 17 | 0 | 74 |
| ios18.leaky_relu | 51 | 0 | 18 | 0 | 69 |
| ios18.add | 0 | 0 | 44 | 0 | 44 |
| ios18.conv_transpose | 3 | 0 | 1 | 0 | 4 |
| ios18.mul | 3 | 0 | 1 | 0 | 4 |
| ios18.tanh | 1 | 0 | 0 | 0 | 1 |

### Transformer_Decoder_float16_fixed262 / Float16 / all

| operator | cpu | gpu | NE | unknown | total |
|---|---:|---:|---:|---:|---:|
| const | 0 | 0 | 0 | 312 | 312 |
| ios18.reshape | 0 | 0 | 48 | 0 | 48 |
| ios18.transpose | 0 | 0 | 32 | 0 | 32 |
| ios18.linear | 0 | 0 | 28 | 0 | 28 |
| ios18.add | 1 | 0 | 11 | 0 | 12 |
| ios18.matmul | 0 | 0 | 12 | 0 | 12 |
| ios18.conv | 0 | 0 | 11 | 0 | 11 |
| ios18.cast | 9 | 0 | 0 | 0 | 9 |
| ios18.layer_norm | 0 | 0 | 9 | 0 | 9 |
| ios18.mul | 0 | 0 | 7 | 0 | 7 |
| ios18.softmax | 0 | 0 | 6 | 0 | 6 |
| ios18.concat | 0 | 0 | 6 | 0 | 6 |
| ios18.relu | 0 | 0 | 5 | 0 | 5 |
| ios18.slice_by_index | 0 | 0 | 5 | 0 | 5 |
| select | 4 | 0 | 0 | 0 | 4 |
| ios18.batch_norm | 0 | 0 | 4 | 0 | 4 |
| ios18.tanh | 0 | 0 | 4 | 0 | 4 |
| ios18.greater_equal | 1 | 0 | 0 | 0 | 1 |
| ios18.gather | 1 | 0 | 0 | 0 | 1 |

### Transformer_Decoder_float16_fixed262 / Float16 / cpuAndGPU

| operator | cpu | gpu | NE | unknown | total |
|---|---:|---:|---:|---:|---:|
| const | 0 | 0 | 0 | 312 | 312 |
| ios18.reshape | 0 | 48 | 0 | 0 | 48 |
| ios18.transpose | 0 | 32 | 0 | 0 | 32 |
| ios18.linear | 0 | 28 | 0 | 0 | 28 |
| ios18.add | 0 | 12 | 0 | 0 | 12 |
| ios18.matmul | 0 | 12 | 0 | 0 | 12 |
| ios18.conv | 0 | 11 | 0 | 0 | 11 |
| ios18.cast | 0 | 9 | 0 | 0 | 9 |
| ios18.layer_norm | 0 | 9 | 0 | 0 | 9 |
| ios18.mul | 0 | 7 | 0 | 0 | 7 |
| ios18.softmax | 0 | 6 | 0 | 0 | 6 |
| ios18.concat | 0 | 6 | 0 | 0 | 6 |
| ios18.relu | 0 | 5 | 0 | 0 | 5 |
| ios18.slice_by_index | 0 | 5 | 0 | 0 | 5 |
| select | 0 | 4 | 0 | 0 | 4 |
| ios18.batch_norm | 0 | 4 | 0 | 0 | 4 |
| ios18.tanh | 0 | 4 | 0 | 0 | 4 |
| ios18.greater_equal | 0 | 1 | 0 | 0 | 1 |
| ios18.gather | 0 | 1 | 0 | 0 | 1 |

### Transformer_Decoder_float16_fixed262 / Float16 / cpuAndNE

| operator | cpu | gpu | NE | unknown | total |
|---|---:|---:|---:|---:|---:|
| const | 0 | 0 | 0 | 312 | 312 |
| ios18.reshape | 0 | 0 | 48 | 0 | 48 |
| ios18.transpose | 0 | 0 | 32 | 0 | 32 |
| ios18.linear | 0 | 0 | 28 | 0 | 28 |
| ios18.add | 1 | 0 | 11 | 0 | 12 |
| ios18.matmul | 0 | 0 | 12 | 0 | 12 |
| ios18.conv | 0 | 0 | 11 | 0 | 11 |
| ios18.cast | 9 | 0 | 0 | 0 | 9 |
| ios18.layer_norm | 0 | 0 | 9 | 0 | 9 |
| ios18.mul | 0 | 0 | 7 | 0 | 7 |
| ios18.softmax | 0 | 0 | 6 | 0 | 6 |
| ios18.concat | 0 | 0 | 6 | 0 | 6 |
| ios18.relu | 0 | 0 | 5 | 0 | 5 |
| ios18.slice_by_index | 0 | 0 | 5 | 0 | 5 |
| select | 4 | 0 | 0 | 0 | 4 |
| ios18.batch_norm | 0 | 0 | 4 | 0 | 4 |
| ios18.tanh | 0 | 0 | 4 | 0 | 4 |
| ios18.greater_equal | 1 | 0 | 0 | 0 | 1 |
| ios18.gather | 1 | 0 | 0 | 0 | 1 |

### Transformer_Decoder_float32_fixed262 / Float32 / all

| operator | cpu | gpu | NE | unknown | total |
|---|---:|---:|---:|---:|---:|
| const | 0 | 0 | 0 | 303 | 303 |
| ios18.reshape | 0 | 48 | 0 | 0 | 48 |
| ios18.transpose | 0 | 32 | 0 | 0 | 32 |
| ios18.linear | 0 | 28 | 0 | 0 | 28 |
| ios18.add | 0 | 12 | 0 | 0 | 12 |
| ios18.matmul | 0 | 12 | 0 | 0 | 12 |
| ios18.conv | 0 | 11 | 0 | 0 | 11 |
| ios18.layer_norm | 0 | 9 | 0 | 0 | 9 |
| ios18.mul | 0 | 7 | 0 | 0 | 7 |
| ios18.softmax | 0 | 6 | 0 | 0 | 6 |
| ios18.concat | 0 | 6 | 0 | 0 | 6 |
| ios18.relu | 0 | 5 | 0 | 0 | 5 |
| ios18.slice_by_index | 0 | 5 | 0 | 0 | 5 |
| select | 0 | 4 | 0 | 0 | 4 |
| ios18.batch_norm | 0 | 4 | 0 | 0 | 4 |
| ios18.tanh | 0 | 4 | 0 | 0 | 4 |
| ios18.greater_equal | 0 | 1 | 0 | 0 | 1 |
| ios18.gather | 0 | 1 | 0 | 0 | 1 |

### Transformer_Decoder_float32_fixed262 / Float32 / cpuAndGPU

| operator | cpu | gpu | NE | unknown | total |
|---|---:|---:|---:|---:|---:|
| const | 0 | 0 | 0 | 303 | 303 |
| ios18.reshape | 0 | 48 | 0 | 0 | 48 |
| ios18.transpose | 0 | 32 | 0 | 0 | 32 |
| ios18.linear | 0 | 28 | 0 | 0 | 28 |
| ios18.add | 0 | 12 | 0 | 0 | 12 |
| ios18.matmul | 0 | 12 | 0 | 0 | 12 |
| ios18.conv | 0 | 11 | 0 | 0 | 11 |
| ios18.layer_norm | 0 | 9 | 0 | 0 | 9 |
| ios18.mul | 0 | 7 | 0 | 0 | 7 |
| ios18.softmax | 0 | 6 | 0 | 0 | 6 |
| ios18.concat | 0 | 6 | 0 | 0 | 6 |
| ios18.relu | 0 | 5 | 0 | 0 | 5 |
| ios18.slice_by_index | 0 | 5 | 0 | 0 | 5 |
| select | 0 | 4 | 0 | 0 | 4 |
| ios18.batch_norm | 0 | 4 | 0 | 0 | 4 |
| ios18.tanh | 0 | 4 | 0 | 0 | 4 |
| ios18.greater_equal | 0 | 1 | 0 | 0 | 1 |
| ios18.gather | 0 | 1 | 0 | 0 | 1 |

### Transformer_Decoder_float32_fixed262 / Float32 / cpuAndNE

| operator | cpu | gpu | NE | unknown | total |
|---|---:|---:|---:|---:|---:|
| const | 0 | 0 | 0 | 303 | 303 |
| ios18.reshape | 48 | 0 | 0 | 0 | 48 |
| ios18.transpose | 32 | 0 | 0 | 0 | 32 |
| ios18.linear | 28 | 0 | 0 | 0 | 28 |
| ios18.add | 12 | 0 | 0 | 0 | 12 |
| ios18.matmul | 12 | 0 | 0 | 0 | 12 |
| ios18.conv | 11 | 0 | 0 | 0 | 11 |
| ios18.layer_norm | 9 | 0 | 0 | 0 | 9 |
| ios18.mul | 7 | 0 | 0 | 0 | 7 |
| ios18.softmax | 6 | 0 | 0 | 0 | 6 |
| ios18.concat | 6 | 0 | 0 | 0 | 6 |
| ios18.relu | 5 | 0 | 0 | 0 | 5 |
| ios18.slice_by_index | 5 | 0 | 0 | 0 | 5 |
| select | 4 | 0 | 0 | 0 | 4 |
| ios18.batch_norm | 4 | 0 | 0 | 0 | 4 |
| ios18.tanh | 4 | 0 | 0 | 0 | 4 |
| ios18.greater_equal | 1 | 0 | 0 | 0 | 1 |
| ios18.gather | 1 | 0 | 0 | 0 | 1 |

### Transformer_Decoder_float32_fixed262 / Float32 / cpuOnly

| operator | cpu | gpu | NE | unknown | total |
|---|---:|---:|---:|---:|---:|
| const | 0 | 0 | 0 | 303 | 303 |
| ios18.reshape | 48 | 0 | 0 | 0 | 48 |
| ios18.transpose | 32 | 0 | 0 | 0 | 32 |
| ios18.linear | 28 | 0 | 0 | 0 | 28 |
| ios18.add | 12 | 0 | 0 | 0 | 12 |
| ios18.matmul | 12 | 0 | 0 | 0 | 12 |
| ios18.conv | 11 | 0 | 0 | 0 | 11 |
| ios18.layer_norm | 9 | 0 | 0 | 0 | 9 |
| ios18.mul | 7 | 0 | 0 | 0 | 7 |
| ios18.softmax | 6 | 0 | 0 | 0 | 6 |
| ios18.concat | 6 | 0 | 0 | 0 | 6 |
| ios18.relu | 5 | 0 | 0 | 0 | 5 |
| ios18.slice_by_index | 5 | 0 | 0 | 0 | 5 |
| select | 4 | 0 | 0 | 0 | 4 |
| ios18.batch_norm | 4 | 0 | 0 | 0 | 4 |
| ios18.tanh | 4 | 0 | 0 | 0 | 4 |
| ios18.greater_equal | 1 | 0 | 0 | 0 | 1 |
| ios18.gather | 1 | 0 | 0 | 0 | 1 |

### Transformer_Decoder_int8_fixed262 / Int8 / all

| operator | cpu | gpu | NE | unknown | total |
|---|---:|---:|---:|---:|---:|
| const | 0 | 0 | 0 | 275 | 275 |
| ios18.reshape | 0 | 0 | 48 | 0 | 48 |
| ios18.constexpr_blockwise_shift_scale | 0 | 0 | 0 | 40 | 40 |
| ios18.transpose | 0 | 0 | 32 | 0 | 32 |
| ios18.linear | 0 | 0 | 28 | 0 | 28 |
| ios18.add | 2 | 0 | 11 | 0 | 13 |
| ios18.matmul | 0 | 0 | 12 | 0 | 12 |
| ios18.conv | 0 | 0 | 11 | 0 | 11 |
| ios18.cast | 10 | 0 | 0 | 0 | 10 |
| ios18.layer_norm | 0 | 0 | 9 | 0 | 9 |
| ios18.mul | 0 | 0 | 7 | 0 | 7 |
| ios18.softmax | 0 | 0 | 6 | 0 | 6 |
| ios18.concat | 0 | 0 | 6 | 0 | 6 |
| ios18.relu | 0 | 0 | 5 | 0 | 5 |
| select | 5 | 0 | 0 | 0 | 5 |
| ios18.slice_by_index | 0 | 0 | 5 | 0 | 5 |
| ios18.batch_norm | 0 | 0 | 4 | 0 | 4 |
| ios18.tanh | 0 | 0 | 4 | 0 | 4 |
| ios18.greater_equal | 2 | 0 | 0 | 0 | 2 |
| ios18.gather | 1 | 0 | 0 | 0 | 1 |

### Transformer_Decoder_int8_fixed262 / Int8 / cpuAndGPU

| operator | cpu | gpu | NE | unknown | total |
|---|---:|---:|---:|---:|---:|
| const | 0 | 0 | 0 | 275 | 275 |
| ios18.reshape | 0 | 48 | 0 | 0 | 48 |
| ios18.constexpr_blockwise_shift_scale | 0 | 0 | 0 | 40 | 40 |
| ios18.transpose | 0 | 32 | 0 | 0 | 32 |
| ios18.linear | 0 | 28 | 0 | 0 | 28 |
| ios18.add | 0 | 13 | 0 | 0 | 13 |
| ios18.matmul | 0 | 12 | 0 | 0 | 12 |
| ios18.conv | 0 | 11 | 0 | 0 | 11 |
| ios18.cast | 0 | 10 | 0 | 0 | 10 |
| ios18.layer_norm | 0 | 9 | 0 | 0 | 9 |
| ios18.mul | 0 | 7 | 0 | 0 | 7 |
| ios18.softmax | 0 | 6 | 0 | 0 | 6 |
| ios18.concat | 0 | 6 | 0 | 0 | 6 |
| ios18.relu | 0 | 5 | 0 | 0 | 5 |
| select | 0 | 5 | 0 | 0 | 5 |
| ios18.slice_by_index | 0 | 5 | 0 | 0 | 5 |
| ios18.batch_norm | 0 | 4 | 0 | 0 | 4 |
| ios18.tanh | 0 | 4 | 0 | 0 | 4 |
| ios18.greater_equal | 0 | 2 | 0 | 0 | 2 |
| ios18.gather | 0 | 1 | 0 | 0 | 1 |

### Transformer_Decoder_int8_fixed262 / Int8 / cpuAndNE

| operator | cpu | gpu | NE | unknown | total |
|---|---:|---:|---:|---:|---:|
| const | 0 | 0 | 0 | 275 | 275 |
| ios18.reshape | 0 | 0 | 48 | 0 | 48 |
| ios18.constexpr_blockwise_shift_scale | 0 | 0 | 0 | 40 | 40 |
| ios18.transpose | 0 | 0 | 32 | 0 | 32 |
| ios18.linear | 0 | 0 | 28 | 0 | 28 |
| ios18.add | 2 | 0 | 11 | 0 | 13 |
| ios18.matmul | 0 | 0 | 12 | 0 | 12 |
| ios18.conv | 0 | 0 | 11 | 0 | 11 |
| ios18.cast | 10 | 0 | 0 | 0 | 10 |
| ios18.layer_norm | 0 | 0 | 9 | 0 | 9 |
| ios18.mul | 0 | 0 | 7 | 0 | 7 |
| ios18.softmax | 0 | 0 | 6 | 0 | 6 |
| ios18.concat | 0 | 0 | 6 | 0 | 6 |
| ios18.relu | 0 | 0 | 5 | 0 | 5 |
| select | 5 | 0 | 0 | 0 | 5 |
| ios18.slice_by_index | 0 | 0 | 5 | 0 | 5 |
| ios18.batch_norm | 0 | 0 | 4 | 0 | 4 |
| ios18.tanh | 0 | 0 | 4 | 0 | 4 |
| ios18.greater_equal | 2 | 0 | 0 | 0 | 2 |
| ios18.gather | 1 | 0 | 0 | 0 | 1 |

### Transformer_Encoder_float16_fixed262 / Float16 / cpuAndNE

| operator | cpu | gpu | NE | unknown | total |
|---|---:|---:|---:|---:|---:|
| const | 0 | 0 | 0 | 154 | 154 |
| ios18.reshape | 0 | 0 | 24 | 0 | 24 |
| ios18.transpose | 0 | 0 | 18 | 0 | 18 |
| ios18.linear | 0 | 0 | 15 | 0 | 15 |
| ios18.add | 1 | 0 | 7 | 0 | 8 |
| ios18.matmul | 0 | 0 | 6 | 0 | 6 |
| ios18.layer_norm | 0 | 0 | 6 | 0 | 6 |
| ios18.conv | 0 | 0 | 6 | 0 | 6 |
| ios18.relu | 0 | 0 | 5 | 0 | 5 |
| ios18.mul | 0 | 0 | 4 | 0 | 4 |
| ios18.cast | 3 | 0 | 0 | 0 | 3 |
| ios18.softmax | 0 | 0 | 3 | 0 | 3 |
| ios18.concat | 0 | 0 | 3 | 0 | 3 |
| ios18.greater_equal | 1 | 0 | 0 | 0 | 1 |
| select | 1 | 0 | 0 | 0 | 1 |
| ios18.gather | 1 | 0 | 0 | 0 | 1 |

### Transformer_Encoder_float32_fixed262 / Float32 / cpuAndGPU

| operator | cpu | gpu | NE | unknown | total |
|---|---:|---:|---:|---:|---:|
| const | 0 | 0 | 0 | 151 | 151 |
| ios18.reshape | 0 | 24 | 0 | 0 | 24 |
| ios18.transpose | 0 | 18 | 0 | 0 | 18 |
| ios18.linear | 0 | 15 | 0 | 0 | 15 |
| ios18.add | 0 | 8 | 0 | 0 | 8 |
| ios18.matmul | 0 | 6 | 0 | 0 | 6 |
| ios18.layer_norm | 0 | 6 | 0 | 0 | 6 |
| ios18.conv | 0 | 6 | 0 | 0 | 6 |
| ios18.relu | 0 | 5 | 0 | 0 | 5 |
| ios18.mul | 0 | 4 | 0 | 0 | 4 |
| ios18.softmax | 0 | 3 | 0 | 0 | 3 |
| ios18.concat | 0 | 3 | 0 | 0 | 3 |
| ios18.greater_equal | 0 | 1 | 0 | 0 | 1 |
| select | 0 | 1 | 0 | 0 | 1 |
| ios18.gather | 0 | 1 | 0 | 0 | 1 |
