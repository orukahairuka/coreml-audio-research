# figures/ ── 研究発表用 図表

PronounSE の iOS 実機評価「インタラクティブ成立条件」を 3軸（速い・壊れない・使える音）で
可視化したスライド用図表。**3軸すべてで同じ ANE 4セル（F16/Int8 × {cpuAndNE, all}）が落ちる**ことを
全図で赤に統一して示す。元レポート: [`docs/2026-06-21/ios-feasibility-evaluation-report.md`](../docs/2026-06-21/ios-feasibility-evaluation-report.md)

> 注: `figures/` 直下の `iPhone_3_*` ディレクトリはデバイスベンチの生成物で git 管理外。
> このスライド用 png/svg と本 README のみ追跡対象（`.gitignore` で制御）。

## 図一覧

| ファイル | 軸 | 内容 |
|---|---|---|
| `feasibility_map.png/.svg` | 統合 | 成立マップ（3軸 ○× マトリクス）。同じ4セルが全軸で不成立 |
| `load_time_initial.png/.svg` | ①速い | 初回モデルロード時間（12セル）。ANE 4セルだけ 36〜63秒/crash |
| `load_time_cached.png/.svg` | ①速い（補足） | キャッシュ後ロード時間。全セル ~0.1〜0.2秒に収束 |
| `mcd_ranking.png/.svg` | ③使える音 | MCD 昇順ランキング。8位↔9位に約5dBギャップ |
| `stability_matrix.png/.svg` | ②壊れない | 安定性 3×4 マトリクス（normal / clipped） |

各図は PNG（150dpi, スライド貼付用）と SVG（ベクター）を同時出力。色は **成立候補=緑 / 不成立=赤** で全図統一。

## 再生成

```bash
PronounSE/venv/bin/python scripts/make_figures.py
```

- スクリプト: [`scripts/make_figures.py`](../scripts/make_figures.py)（matplotlib）
- 日本語フォント: `ヒラギノ角ゴシック W3.ttc` を自動検出。無ければ英語ラベルにフォールバック

## 元データ（`data/2026-06-21/figures/`）

CSV を編集して再実行すれば図が更新される。値の出典は以下。

| CSV | 図 | 出典ドキュメント |
|---|---|---|
| `load_time.csv` | 図2・図3 | [load-timing-results](../docs/2026-06-10/load-timing-results.md)（iPhone(3), 2026-06-10, n=2）。初回は run1/run2 平均。F16×all は run1 がクラッシュのため run2 値（36.69s）を使用し crash 注記 |
| `mcd.csv` | 図4 | [audio-quality-mcd-results](../docs/2026-06-21/audio-quality-mcd-results.md)（MCD dtw, 対 PyTorch baseline, n=1）|
| `stability.csv` | 図5 | [stability-matrix-analysis](../docs/2026-05-19/stability-matrix-analysis.md) ＋ [ane-stage-isolation-breakdown](../docs/2026-06-21/ane-stage-isolation-breakdown.md) |
| `feasibility.csv` | 図1 | 上記3軸の統合（[ios-feasibility-evaluation-report](../docs/2026-06-21/ios-feasibility-evaluation-report.md) §8 成立マップ）|

### 主要な数値（参考）

- **初回ロード**: CPU/GPU系 0.66〜1.22s、ANE系 36.06〜62.71s（F16 all は run1 crash→36.69s）
- **キャッシュ後**: 全セル 77〜196ms
- **MCD**: CPU/GPU系8セル 3.92〜6.63dB、ANE系4セル 11.61〜12.02dB（約5dBギャップ）
- **安定性**: F16/Int8 × {cpuAndNE, all} の4セルのみ clipped、他8セルは normal

> n=1 の軸（MCD・安定性）と n=2 の軸（ロード時間）が混在。8セル内部の順位は未確定で、
> robust なのは「8 vs 4 の段差」。詳細は各元ドキュメントの残課題を参照。
