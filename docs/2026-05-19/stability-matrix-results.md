# Phase 1 — 全 12 組合せ安定性マトリクス結果

集計対象: 78 runs, 取り込み元 1 archive

生成元スクリプト: `scripts/aggregate_stability_matrix.py`

## 1. 分類別カウント（precision × computeUnit）

| precision | computeUnit | normal_loud | quiet | clipped | nan_inf | predict_failed |
|---|---|---:|---:|---:|---:|---:|
| Float16 | all | 0 | 0 | 3 | 0 | 0 |
| Float16 | cpuAndGPU | 6 | 0 | 0 | 0 | 1 |
| Float16 | cpuAndNE | 0 | 0 | 3 | 0 | 0 |
| Float16 | cpuOnly | 3 | 0 | 0 | 0 | 0 |
| Float32 | all | 7 | 0 | 0 | 0 | 0 |
| Float32 | cpuAndGPU | 14 | 12 | 4 | 0 | 0 |
| Float32 | cpuAndNE | 3 | 0 | 0 | 0 | 0 |
| Float32 | cpuOnly | 7 | 0 | 0 | 0 | 0 |
| Int8 | all | 0 | 0 | 3 | 0 | 0 |
| Int8 | cpuAndGPU | 6 | 0 | 0 | 0 | 0 |
| Int8 | cpuAndNE | 0 | 0 | 3 | 0 | 0 |
| Int8 | cpuOnly | 3 | 0 | 0 | 0 | 0 |

## 2. iter 別ヒートマップ（precision × computeUnit × iter）

値は最多分類を示す。複数同数なら `?`。

| precision | computeUnit | iter1 | iter2 | iter3 | iter10 |
|---|---|---|---|---|---|
| Float16 | all | clipped | clipped | clipped | - |
| Float16 | cpuAndGPU | normal_loud | normal_loud | normal_loud | - |
| Float16 | cpuAndNE | clipped | clipped | clipped | - |
| Float16 | cpuOnly | normal_loud | normal_loud | normal_loud | - |
| Float32 | all | normal_loud | normal_loud | normal_loud | - |
| Float32 | cpuAndGPU | ? | normal_loud | normal_loud | quiet |
| Float32 | cpuAndNE | normal_loud | normal_loud | normal_loud | - |
| Float32 | cpuOnly | normal_loud | normal_loud | normal_loud | - |
| Int8 | all | clipped | clipped | clipped | - |
| Int8 | cpuAndGPU | normal_loud | normal_loud | normal_loud | - |
| Int8 | cpuAndNE | clipped | clipped | clipped | - |
| Int8 | cpuOnly | normal_loud | normal_loud | normal_loud | - |

## 3. NaN/Inf を含む run の Decoder step1 統計

（NaN/Inf を含む run はなし）

## 4. 個別 run 一覧

| precision | computeUnit | iter | class | rms | peak | postnet sha | run |
|---|---|---|---|---|---|---|---|
| Float16 | all | 1 | clipped | 13284.4 | 68336.4 | dc8df52b33 | f16AllRepeat1 |
| Float16 | all | 2 | clipped | 13284.4 | 68336.4 | dc8df52b33 | f16AllRepeat2 |
| Float16 | all | 3 | clipped | 13284.4 | 68336.4 | dc8df52b33 | f16AllRepeat3 |
| Float16 | cpuAndGPU | 1 | normal_loud | 4993.2 | 24176.4 | a8fabfd8d1 | f16GpuRepeat1 |
| Float16 | cpuAndGPU | 1 | predict_failed |  |  | - | f16GpuRepeat1 |
| Float16 | cpuAndGPU | 1 | normal_loud | 4993.2 | 24176.4 | a8fabfd8d1 | f16GpuRepeat1 |
| Float16 | cpuAndGPU | 2 | normal_loud | 4993.2 | 24176.4 | a8fabfd8d1 | f16GpuRepeat2 |
| Float16 | cpuAndGPU | 2 | normal_loud | 4993.2 | 24176.4 | a8fabfd8d1 | f16GpuRepeat2 |
| Float16 | cpuAndGPU | 3 | normal_loud | 4993.2 | 24176.4 | a8fabfd8d1 | f16GpuRepeat3 |
| Float16 | cpuAndGPU | 3 | normal_loud | 4993.2 | 24176.4 | a8fabfd8d1 | f16GpuRepeat3 |
| Float16 | cpuAndNE | 1 | clipped | 13289.5 | 71468.9 | dc8df52b33 | f16NeRepeat1 |
| Float16 | cpuAndNE | 2 | clipped | 13289.5 | 71468.9 | dc8df52b33 | f16NeRepeat2 |
| Float16 | cpuAndNE | 3 | clipped | 13289.5 | 71468.9 | dc8df52b33 | f16NeRepeat3 |
| Float16 | cpuOnly | 1 | normal_loud | 4911.9 | 25432.5 | 492d021982 | f16CpuRepeat1 |
| Float16 | cpuOnly | 2 | normal_loud | 4911.9 | 25432.5 | 492d021982 | f16CpuRepeat2 |
| Float16 | cpuOnly | 3 | normal_loud | 4911.9 | 25432.5 | 492d021982 | f16CpuRepeat3 |
| Float32 | all | None | normal_loud | 3233.7 | 28294.7 | 28a611865b | freshFirstAll |
| Float32 | all | 1 | normal_loud | 5029.1 | 24326.0 | 23c51a5431 | f32AllRepeat1 |
| Float32 | all | 1 | normal_loud | 5029.1 | 24326.0 | 23c51a5431 | f32AllRepeat1 |
| Float32 | all | 2 | normal_loud | 5029.1 | 24326.0 | 23c51a5431 | f32AllRepeat2 |
| Float32 | all | 2 | normal_loud | 5029.1 | 24326.0 | 23c51a5431 | f32AllRepeat2 |
| Float32 | all | 3 | normal_loud | 5029.1 | 24326.0 | 23c51a5431 | f32AllRepeat3 |
| Float32 | all | 3 | normal_loud | 5029.1 | 24326.0 | 23c51a5431 | f32AllRepeat3 |
| Float32 | cpuAndGPU | None | quiet | 711.9 | 23873.6 | 483ab4e454 | freshFirstNoSleep |
| Float32 | cpuAndGPU | None | normal_loud | 5029.1 | 24326.0 | 23c51a5431 | warmupDummy |
| Float32 | cpuAndGPU | None | normal_loud | 5029.1 | 24326.0 | 23c51a5431 | warmupReal |
| Float32 | cpuAndGPU | None | quiet | 945.0 | 24501.5 | 6cecb13ab9 | freshFirstNoSleep |
| Float32 | cpuAndGPU | None | quiet | 1441.0 | 28798.2 | 8cc9788ddc | freshFirstNoSleep |
| Float32 | cpuAndGPU | None | quiet | 1443.6 | 28798.2 | 75e0a50212 | freshFirstNoSleep |
| Float32 | cpuAndGPU | None | clipped | 8441.4 | 33162.5 | 455c1659ba | warmupDummy |
| Float32 | cpuAndGPU | None | normal_loud | 5029.1 | 24326.0 | 23c51a5431 | warmupReal |
| Float32 | cpuAndGPU | 1 | quiet | 900.8 | 26008.0 | fa3026262c | freshFirstRepeat1 |
| Float32 | cpuAndGPU | 1 | quiet | 750.1 | 11518.5 | 4483eba938 | directRepeat1 |
| Float32 | cpuAndGPU | 1 | quiet | 2121.0 | 24986.3 | 12f658502f | directRepeat1 |
| Float32 | cpuAndGPU | 1 | clipped | 8441.4 | 33162.5 | 455c1659ba | directRun1 |
| Float32 | cpuAndGPU | 1 | normal_loud | 5029.1 | 24326.0 | 23c51a5431 | directRepeat1 |
| Float32 | cpuAndGPU | 1 | clipped | 8441.4 | 33162.5 | 455c1659ba | directRun1 |
| Float32 | cpuAndGPU | 1 | normal_loud | 5029.1 | 24326.0 | 23c51a5431 | directRepeat1 |
| Float32 | cpuAndGPU | 1 | clipped | 8441.4 | 33162.5 | 455c1659ba | directRun1 |
| Float32 | cpuAndGPU | 1 | normal_loud | 5029.1 | 24326.0 | 23c51a5431 | directRepeat1 |
| Float32 | cpuAndGPU | 2 | quiet | 861.0 | 24799.3 | d869e3caf8 | freshFirstRepeat2 |
| Float32 | cpuAndGPU | 2 | normal_loud | 5029.1 | 24326.0 | 23c51a5431 | directRepeat2 |
| Float32 | cpuAndGPU | 2 | normal_loud | 5029.1 | 24326.0 | 23c51a5431 | directRepeat2 |
| Float32 | cpuAndGPU | 2 | normal_loud | 5029.1 | 24326.0 | 23c51a5431 | directRepeat2 |
| Float32 | cpuAndGPU | 2 | normal_loud | 5029.1 | 24326.0 | 23c51a5431 | directRepeat2 |
| Float32 | cpuAndGPU | 3 | quiet | 861.0 | 24799.3 | d869e3caf8 | freshFirstRepeat3 |
| Float32 | cpuAndGPU | 3 | normal_loud | 5029.1 | 24326.0 | 23c51a5431 | directRepeat3 |
| Float32 | cpuAndGPU | 3 | normal_loud | 5029.1 | 24326.0 | 23c51a5431 | directRepeat3 |
| Float32 | cpuAndGPU | 3 | normal_loud | 5029.1 | 24326.0 | 23c51a5431 | directRepeat3 |
| Float32 | cpuAndGPU | 3 | normal_loud | 5029.1 | 24326.0 | 23c51a5431 | directRepeat3 |
| Float32 | cpuAndGPU | 10 | quiet | 711.9 | 23873.6 | 483ab4e454 | freshFirstSleep10 |
| Float32 | cpuAndGPU | 10 | quiet | 611.7 | 15865.6 | 4a8ee69576 | freshFirstSleep10 |
| Float32 | cpuAndGPU | 10 | quiet | 945.0 | 24501.5 | 6cecb13ab9 | freshFirstSleep10 |
| Float32 | cpuAndNE | 1 | normal_loud | 5029.1 | 24326.4 | 56ed1e49d0 | f32NeRepeat1 |
| Float32 | cpuAndNE | 2 | normal_loud | 5029.1 | 24326.4 | 56ed1e49d0 | f32NeRepeat2 |
| Float32 | cpuAndNE | 3 | normal_loud | 5029.1 | 24326.4 | 56ed1e49d0 | f32NeRepeat3 |
| Float32 | cpuOnly | None | normal_loud | 3285.2 | 23466.8 | d3b18f1621 | freshFirstCpuOnly |
| Float32 | cpuOnly | 1 | normal_loud | 5029.1 | 24326.4 | 56ed1e49d0 | f32CpuRepeat1 |
| Float32 | cpuOnly | 1 | normal_loud | 5029.1 | 24326.4 | 56ed1e49d0 | f32CpuRepeat1 |
| Float32 | cpuOnly | 2 | normal_loud | 5029.1 | 24326.4 | 56ed1e49d0 | f32CpuRepeat2 |
| Float32 | cpuOnly | 2 | normal_loud | 5029.1 | 24326.4 | 56ed1e49d0 | f32CpuRepeat2 |
| Float32 | cpuOnly | 3 | normal_loud | 5029.1 | 24326.4 | 56ed1e49d0 | f32CpuRepeat3 |
| Float32 | cpuOnly | 3 | normal_loud | 5029.1 | 24326.4 | 56ed1e49d0 | f32CpuRepeat3 |
| Int8 | all | 1 | clipped | 14162.9 | 83202.4 | 0b61fb6512 | int8AllRepeat1 |
| Int8 | all | 2 | clipped | 14162.9 | 83202.4 | 0b61fb6512 | int8AllRepeat2 |
| Int8 | all | 3 | clipped | 14162.9 | 83202.4 | 0b61fb6512 | int8AllRepeat3 |
| Int8 | cpuAndGPU | 1 | normal_loud | 5534.3 | 27709.9 | 33c43a8790 | int8GpuRepeat1 |
| Int8 | cpuAndGPU | 1 | normal_loud | 5534.3 | 27709.9 | 33c43a8790 | int8GpuRepeat1 |
| Int8 | cpuAndGPU | 2 | normal_loud | 5534.3 | 27709.9 | 33c43a8790 | int8GpuRepeat2 |
| Int8 | cpuAndGPU | 2 | normal_loud | 5534.3 | 27709.9 | 33c43a8790 | int8GpuRepeat2 |
| Int8 | cpuAndGPU | 3 | normal_loud | 5534.3 | 27709.9 | 33c43a8790 | int8GpuRepeat3 |
| Int8 | cpuAndGPU | 3 | normal_loud | 5534.3 | 27709.9 | 33c43a8790 | int8GpuRepeat3 |
| Int8 | cpuAndNE | 1 | clipped | 14154.9 | 87399.9 | 0b61fb6512 | int8NeRepeat1 |
| Int8 | cpuAndNE | 2 | clipped | 14154.9 | 87399.9 | 0b61fb6512 | int8NeRepeat2 |
| Int8 | cpuAndNE | 3 | clipped | 14154.9 | 87399.9 | 0b61fb6512 | int8NeRepeat3 |
| Int8 | cpuOnly | 1 | normal_loud | 5607.9 | 28170.8 | 1803ce7f9b | int8CpuRepeat1 |
| Int8 | cpuOnly | 2 | normal_loud | 5607.9 | 28170.8 | 1803ce7f9b | int8CpuRepeat2 |
| Int8 | cpuOnly | 3 | normal_loud | 5607.9 | 28170.8 | 1803ce7f9b | int8CpuRepeat3 |
