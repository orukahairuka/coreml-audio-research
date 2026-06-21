# 音質（使える音）軸 MCD 計測 ── 対 PyTorch baseline

12セル（精度3 × 計算デバイス4, shape=fixed262）の出力が、**変換前の PyTorch 出力（baseline）から
どれだけ音色が劣化したか**を MCD（メルケプストラム歪み）で計測した。
「その場でインタラクティブ」の成立を問う [feasibility study](../2026-06-08/interactive-feasibility-direction.md)
の3軸目「**使える音**」を埋めるのが目的。

- **指標**: MCD（mode=dtw）。2音をフレームごとにメルケプストラム係数へ変換し、DTW で時刻を揃えて
  距離を平均した値[dB]。小さいほど baseline に音色が近い。`pymcd` で算出。
- **ハーネス**: [`scripts/compare_mcd.py`](../../scripts/compare_mcd.py)（`--ref` で baseline 差し替え可）
- **入力**: `input_sample.wav`。iOS アプリがバンドルする入力（`SynthesisViewModel.swift:154`）と
  byte-identical（sha256 一致）なので、実機セル出力と同一入力であることを確認済み。
- **baseline**: `PronounSE/synthesis.py input_sample.wav` を 2026-06-21 に再生成した PyTorch 出力
  （`input_sample_synth.wav`, 22050Hz, 3.09s）。旧版（Mar 21）と byte size 一致＝パイプライン出力は安定。
- **対象セル**: `result/output_{精度}_{デバイス}.wav` の12個（2026-05-16〜19 の実機 iPhone(3) スイープ）

## なぜ baseline が PyTorch 出力なのか

本研究の問いは「**PC（PyTorch）でできていたことがスマホ（CoreML）で同じように成立するか**」。
したがって「劣化しなかったら本来どう鳴るはずか」の基準＝ PyTorch 出力に置く。同じ入力 `input_sample.wav`
を PyTorch と CoreML 各セルに食わせ、PyTorch 出力を正解として MCD を測る。教師音（親研究の正解爆発音）
との絶対比較は別軸で、ここでは扱わない。

## 結果

| 順位 | セル | MCD(dB) | 判定 |
|---:|------|--------:|:---:|
| 1 | Float32 × all | 3.92 | 🟢 |
| 2 | Float32 × cpuAndGPU | 5.21 | 🟢 |
| 3 | Float32 × cpuOnly | 5.35 | 🟢 |
| 4 | Int8 × cpuAndGPU | 5.93 | 🟢 |
| 5 | Float32 × cpuAndNE | 6.58 | 🟢 |
| 6 | Float16 × cpuAndGPU | 6.59 | 🟢 |
| 7 | Int8 × cpuOnly | 6.60 | 🟢 |
| 8 | Float16 × cpuOnly | 6.63 | 🟢 |
| | **── 約 5dB のギャップ ──** | | |
| 9 | **Float16 × all** | **11.61** | 🔴 |
| 10 | **Float16 × cpuAndNE** | **11.84** | 🔴 |
| 11 | **Int8 × cpuAndNE** | **11.99** | 🔴 |
| 12 | **Int8 × all** | **12.02** | 🔴 |

## 読み取れること

### 1. 「使える音」軸も同じ ANE 4セルを棄却。境界は対 PyTorch でより鋭い

- **CPU/GPU 系8セル = 3.9〜6.6 dB に集中**。これが「CoreML 変換の素の劣化幅」。
- **ANE 系4セル（F16/Int8 × {cpuAndNE, all}）= 11.6〜12.0 dB に隔離**。8セルの最悪(6.6)との間に
  **約 5dB の空白**があり、明確な境界線が引ける。
- → [clipping で「壊れない」軸が落ちた4セル](../2026-05-19/stability-matrix-analysis.md)、
  [初回コンパイルで「速い」軸が落ちた4セル](../2026-06-10/load-timing-results.md) と**完全一致**。
  **3軸すべてが同じ ANE 4セルを同時棄却**した。dispatch 上 HiFi-GAN が NE に乗る構成
  （[dispatch マップ](../2026-05-19/compute-plan-analysis.md)）で、振幅破綻が MCD にも素直に出ている。

### 2. MCD の床は ~4dB（0 ではない）

最良の F32×all でも 3.92 dB。CoreML 変換は F32 でも PyTorch と完全一致はせず、加えて独立生成した
2波形の DTW アライメント差が乗るため、これが「動くセルの下限」。0 を期待するものではない。

### 3. 8セル内の順位は n=1 でまだ確定できない

F32×all が1位・F32×cpuOnly が3位といった**8セル内部の細かい順位は、入力1本＋実機の初回非決定性
（[既知](../2026-05-17/fp32-cpuandgpu-quiet-vs-loud-investigation.md)）の揺れの範囲**で、
「どのセルが最良か」はこのデータでは決められない。robust なのは「8 vs 4 の約5dB段差」の方。

> 補足: F32×{cpuAndNE, all} が8セル側（良）にいるのは、[F32 が ANE 非対応](../2026-05-19/stability-matrix-analysis.md)
> で NE に dispatch されず CPU/GPU にフォールバックするため。"ANE 指定でも実際は ANE を使っていない"。

## 成立マップへの含意

3軸が初めて出そろい、すべて同じ結論に収束した：

| 設定 | ①速い | ②壊れない | ③使える音(MCD) | 成立 |
|---|:---:|:---:|:---:|:---:|
| F32/F16/Int8 × {cpuOnly, cpuAndGPU} | 〇 ~1s | 〇 | 〇 3.9〜6.6 | **成立候補** |
| F32 × {cpuAndNE, all}（実態CPU/GPU） | 〇 ~1s | 〇 | 〇 6.6 | 成立候補 |
| **F16/Int8 × {cpuAndNE, all}** | ✕ 32-67s/crash | ✕ clipping | ✕ 11.6+ | **不成立** |

→ インタラクティブ成立の候補は **CPU/GPU 側8セル**に絞られる、を3軸独立に裏づけた。

## 残課題

- **入力を増やして8セル内の順位を傾向化**（n=1 脱却。先輩データセットの声真似 wav が候補）。
  破綻4セルの棄却は確定だが、候補8セルの優劣はまだ「たまたま」を排除できていない。
- **上位数セルの聴感確認（ABX/聴感）**。MCD は足切りで、最終判断は耳。MCD 4〜6.6dB の差が
  人に知覚されるか（特に量子化 F16/Int8 が F32 とどれだけ違って聞こえるか）は別途。
- **救済策（J 正規化 / K GPU 退避）後の MCD**。破綻4セルを救った音が baseline にどこまで近づくか。
