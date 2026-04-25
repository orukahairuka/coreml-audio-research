# 研究の前提 （2026-04-22）

先生相談用と自分の備忘を兼ねた 1 枚。前半が「ここまで固まったこと」、後半が「相談したい論点」。

## 目的

PronounSE（2 段音声合成モデル）を iOS + CoreML で実用的に動かすための基礎づくり。

## 完了していること

- **CoreML 変換 9 モデル**: Encoder / Decoder / HiFi-GAN × Float32 / Float16 / Int8 を一通り変換
  - [transformer-conversion](../2026-03-21/transformer-conversion.md) / [hifigan-conversion](../2026-03-21/hifigan-conversion.md)
- **Python での合成パイプライン検証**: 3 モデルを繋いで音が出るところまで確認
  - [coreml-pipeline](../2026-03-21/coreml-pipeline.md)
- **iOS アプリ（CoreMLAudioApp）で手動合成・可視化**: ファイルを選んで合成→波形・メルの可視化まで動く
  - [ios-app-implementation](../2026-04-05/ios-app-implementation.md) / [models-refactor](../2026-04-17/models-refactor.md) / [decoder-runner-breakdown](../2026-04-22/decoder-runner-breakdown.md)
- **CoreML のラップ・型変換を集計**: 1 秒合成あたり約 114 万回発生していることを定量化
  - [coreml-wrapping-analysis](../2026-04-22/coreml-wrapping-analysis.md)
- **Float32 × HiFi-GAN で音が出ない現象を観察**: GPU / ANE 実行時に波形が崩れる。原因未特定
  - [float32-gpu-debug-report](../2026-04-05/float32-gpu-debug-report.md)

## 固定している前提

- PronounSE は爆発音のみ対応（モデル拡張は研究範囲外）
- iOS 一択（CoreML が研究の核）
- PronounSE は博士の先輩が開発したモデルを使用（submodule）

---

## 先生に相談したい論点

### 論点 1: Phase 1 の測定範囲

**決まっていること**: 3 精度（FP32 / FP16 / Int8）× 4 デバイス（cpuOnly / cpuAndGPU / cpuAndNeuralEngine / all）をベースに測る。

**迷い**: モデル 3 つ（Encoder / Decoder / HiFi-GAN）それぞれに精度を独立に振ると **3³ × 4 = 108 通り**、均一精度だと **12 通り**。

**仮案**: 108 通り全部測定し、考察は代表点（10〜20 個）に絞る。インフラさえ作れば手数は同じ。

**先生に聞きたい**: この範囲で妥当か、絞るべきか / 広げるべきか

---

### 論点 2: 音質指標と基準の取り方

**背景**: PronounSE は **爆発音**（非 speech）。speech 向け指標（PESQ / STOI）は相性が悪い。

**仮案**: **MCD + FAD の 2 本立て**、基準は **Float32 CoreML の出力**

**先生に聞きたい**: 爆発音の評価指標として何が適切か、基準の取り方はこれでいいか

#### 代表的な音質指標（先生から名前が出てきた時の戻り用）

| 指標 | 何を測る | 相性 |
|---|---|---|
| **MCD** (Mel Cepstral Distortion) | メルケプストラム距離、スペクトル距離の古典 | ◎ 爆発音にも通用 |
| **FAD** (Fréchet Audio Distance) | 深層モデル特徴量の分布距離、効果音生成の定番 | ◎ 爆発音向き |
| **PESQ** | 電話音声の知覚品質、ITU-T P.862 | ✗ speech 専用 |
| **STOI** | 音声の明瞭度 | ✗ 言語音声前提 |
| **SNR / SI-SDR** | 波形の信号対雑音比 | △ HiFi-GAN は位相ランダムで荒れる |
| **MOS** | 5 段階絶対評価（主観） | スコープ外（被験者実験） |
| **ABX test** | A / B のどちらが参照に近いか（主観・比較） | 研究室内ミニ実験なら可 |

#### 基準（ground truth）の選択肢

- **Float32 CoreML 出力**（仮案） — 量子化劣化だけ切り出せる
- **PyTorch 出力** — 変換劣化まで含めて見えるが、Float32 CoreML の既存問題で使いにくい面あり

---

### 論点 3: Float32 × HiFi-GAN × GPU/ANE で音が崩れる現象

**観察**: Float32 + GPU / ANE 実行で HiFi-GAN の出力波形が崩れる。CPU では正常。詳細は [float32-gpu-debug-report](../2026-04-05/float32-gpu-debug-report.md)。

**仮説レベルで疑っていること**:
- Metal シェーダ / ANE の内部演算精度が Float32 を丸めている
- CoreML が暗黙に Float16 にキャストしている
- モデル内部の reduce 系演算（LayerNorm 等）で桁落ち

**仮案**: **この現象の原因究明を研究貢献の柱の一つにする**。「CoreML で Float32 指定しても実効精度が保証されない」は実装者向けの知見になる。回避策まで踏み込めれば設計指針に直結。

**先生に聞きたい**:
- この方向で研究の柱に据えていいか
- 原因特定に時間を溶かすリスクがある。どこで切り上げるか
- Apple への Feedback 提出は研究貢献として評価されるか

---

### 論点 4: 研究としての成立性と貢献の重心（根本論点）

**率直な疑問**: **これって「研究」と呼べる？** 「量子化してベンチマーク取りました」だけでは単なる測定作業。学部卒研として成立するか。

**仮案**: 卒研の核を以下の 3 本立てで構成する
1. **CoreML 上での量子化パレート** — 3 精度 × 4 デバイスの実データ + 混合精度探索
2. **失敗原因究明**（論点 3） — Float32 × GPU/ANE 異常挙動の切り分け + 回避策
3. **オンデバイス設計指針** — 1 + 2 から導く具体的ガイド

既存研究は PyTorch / TFLite 中心で、CoreML 特有のデバイス選択肢を組み合わせた網羅データは少ない。この 3 本立てが新規性の在処、と想定している。

**先生に聞きたい**:
- この 3 本立てで「研究」として認めてもらえるか
- 新規性はどこにあると主張すべきか
- 不足しているとしたら何を足せばいいか（関連研究サーベイ、理論的裏付け 等）
- パレート分析と失敗原因究明、重心をどちらに置くべきか
