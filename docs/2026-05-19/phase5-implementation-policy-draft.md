# Phase 5 実装方針 ドラフト（2026-05-19）

ブランチ: `feature/fp32-gpu-memory-timing`
状態: **ドラフト**。確定ではない。Phase 2 mini の音質聴感判定（残課題）と必要なら追試で確定させる。

関連:
- [`all-engine-precision-stability-plan.md`](all-engine-precision-stability-plan.md) §6（Phase 5 — 実装方針の確定）
- [`stability-matrix-analysis.md`](stability-matrix-analysis.md)（Phase 1 観測）
- [`compute-plan-analysis.md`](compute-plan-analysis.md)（Phase 4 dispatch）
- [`phase2-mini-rescue-results.md`](phase2-mini-rescue-results.md)（Phase 2 mini 救済実験）

## 0. 当面の方針メモ（2026-05-19）

**当面は救済戦略（A〜K）を本番側に組み込まない**。

理由: いまは「各 engine × precision を素のまま比較する」フェーズで、warm-up / retry / fallback / 出力正規化 / HiFi-GAN 別エンジン退避といった介入を入れると、計測したい数値（rms / peak / sha256 / 中間 tensor）に戦略の影響が混ざるため。

- 比較したい軸: precision (F32 / F16 / Int8) × computeUnits (cpuOnly / cpuAndGPU / cpuAndNE / all) の 12 セル。
- 各セルは「ロードして synthesize を呼ぶ」までの最短経路で測る。
- 救済策の効果は引き続き別途 (Phase2RescueTests の戦略テスト) で計測し、観測整理に残す。
- 「本番で UI から外す / 残す」「fallback chain を入れる」などは比較が終わってから決める。

要するに **「素の比較を完走させるまで、本番コードに新しい救済を組み込まない」**。

## 1. 確定している観測（数値・sha レベルで再現）

| 観測 | 出典 |
|---|---|
| F32 × cpuOnly / cpuAndNE / all は warm 状態で manual baseline (rms 5029) と一致 | Phase 1 |
| F32 × cpuAndGPU は 1st call 非決定的、2nd call 以降は manual baseline と一致 | Phase 1, 2026-05-17 メモ |
| F16 × cpuAndGPU は 1st call から決定論的に normal_loud (rms ~5040) | Phase 1 |
| F16 / Int8 × cpuOnly / cpuAndGPU は決定論的に normal_loud | Phase 1 |
| F16 / Int8 × cpuAndNE / all は決定論的に clipped (rms 13000+, peak 70000+) | Phase 1 |
| F16 / Int8 × cpuAndNE Repeat3 iter1/2/3 はすべて bit-identical | Phase 1 |
| HiFi-GAN F16 × cpuAndNE で `add` 全 44 個と `conv` 17/74 が NE 行き | Phase 4 |
| K (Decoder=ANE, HiFi-GAN=GPU) の postnet sha は cpuAndNE 単独と bit-identical | Phase 2 mini |
| K の最終波形は normal_loud (F16 rms 4994 / peak 25974, Int8 rms 5431 / peak 27448) | Phase 2 mini |

## 2. 設定別の方針案

各セルは現時点の暫定評価。最終確定は次節の残課題が解けてから。

| precision × computeUnits | Phase 1 結果 | 救済策の見込み | 現時点の方針案 |
|---|---|---|---|
| F32 × cpuOnly | normal_loud | 不要 | 採用候補 |
| F32 × cpuAndGPU | 1st call 非決定的、2nd 以降 loud | warm-up 1 回（Strategy A） | warm-up 込みで採用候補 |
| F32 × cpuAndNE | normal_loud（実態は CPU fallback） | 不要 | 採用候補。ただし「ANE で動いている」という表記はしない |
| F32 × all | normal_loud（実態は CPU/GPU） | 不要 | 採用候補。同上 |
| F16 × cpuOnly | normal_loud | 不要 | 採用候補 |
| F16 × cpuAndGPU | normal_loud | 不要 | **標準採用候補**（既存本番、PyTorch reference に近い） |
| F16 × cpuAndNE | clipped | (J) 出力正規化で振幅は救える / (K) HiFi-GAN を GPU に逃がせば normal_loud | 本番方針は未定。当面は UI に残す。研究設定として K と組み合わせるなら採用可 |
| F16 × all | clipped | 同上 | 本番方針は未定。当面は UI に残す |
| Int8 × cpuOnly | normal_loud | 不要 | 採用候補 |
| Int8 × cpuAndGPU | normal_loud | 不要 | 採用候補 |
| Int8 × cpuAndNE | clipped | (J) (K) F16 と同じ傾向 | 本番方針は未定。当面は UI に残す |
| Int8 × all | clipped | 同上 | 本番方針は未定。当面は UI に残す |

clipping を観測した構成についても、本番でどう扱うかは未定。方針が決まるまで **どのセルも UI から外さず、12 セル全てを選択可能なまま残す**。F32 の cpuAndNE / all は dispatch 上 NE には行かない点も観測として記録しておくが、これも UI からの除外対象ではない。

## 3. 実装提案項目（暫定）

| 項目 | 通常 | 研究計測モード |
|---|---|---|
| 標準 precision | F16 | UI で全 precision 選択可 |
| 標準 computeUnits | cpuAndGPU | UI で全 4 種選択可 |
| warm-up | F32 × cpuAndGPU のときだけ 1 回 | 0 固定（観測したいので） |
| 異常検出 | NaN/Inf、rms < 3000、peak > 32000 を検出してログ | フラグだけ、retry なし |
| retry | しない（warm-up で十分） | 0 |
| fallback chain | しない（明示的な切替のみ） | なし |

retry / fallback chain を入れない理由: Phase 1 で「F16/Int8 × cpuAndNE は決定論的に clipped」なので、同じ設定で retry しても結果は変わらない。fallback も「ユーザーが選んだ設定を勝手に変える」ことになり、研究用 UI の透明性を損なう。clipping する構成を本番でどう扱うかは未定で、方針が決まるまで UI からは外さない。

## 4. 残課題（方針確定の前提）

1. **Phase 2 mini J の音質聴感判定** — `audio/iPhone_3_phase2mini_20260519/playable/*phase2_J_*` の wav を試聴し、正規化で音が救えるかを確認する。
2. **Phase 2 mini K の音質聴感判定** — `*phase2_K_*` の wav を Phase 1 F16 × cpuAndGPU の wav と聴き比べる。同等なら K は確固たる救済策。
3. **本番での扱いの最終決定** — clipping を観測した構成（F16/Int8 × cpuAndNE / all）を本番でどう扱うかは未定。方針が決まるまで UI からは外さず、全セルを選択可能なまま残す。決定は Phase 2 mini の聴感判定（J/K）が済んでから行う。
4. **異常検出ログの実装** — `AudioSynthesizer.synthesize()` の末尾に既にある production log（rms/peak/NaN/Inf）はそのままで足りる。`SynthesisResult` を返した側で classification を出すかどうかは別途検討。

## 5. このドラフトの位置付け

- 現時点では「これで本番に出して良い」と言える根拠は揃っていない。
- 観測した範囲では F16 × cpuAndGPU を標準にする方向は Phase 1 / 2 mini / 4 の観測と整合するが、それ以外のセルを本番から外すかどうかは未定。方針が決まるまではどのセルも UI から外さない。
- 残課題の (1) (2) は耳で聴く検証なので、ユーザー判定待ち。
- (3) (4) はコード変更を伴うので、Phase 5 確定後に別タスクで進める。
