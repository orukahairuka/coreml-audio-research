# 先生相談後の TODO メモ （2026-04-22）

[research-plan](research-plan.md) の論点を先生と相談した結果の持ち帰り事項。

## Float32 × GPU/ANE の非対応問題

- Apple の ANE / GPU は Float32 を対応していない（前提として確定）
- OS を変えて挙動が変わるか試す？（GPU 問題の切り分け候補）, macOSにしてみる
- 参考: [float32-gpu-debug-report](../2026-04-05/float32-gpu-debug-report.md)

## HiFi-GAN の差し替え検証

- いま使っている HiFi-GAN は配布されているモデル
- HiFi-GAN だけ別モデルに差し替えて動くか確認する
- 候補: `hifigan-base`

## 比較・評価

- 指標としてメルスペクトログラムの画像を比較する
- GPU 実行時の出力と比較する
- 行列レベルで比較する
- 熊木さんが差分を取っている → 手法を参考にする

## データ・文献

- 入力データをもらう
- 滝沢さんの論文を読む

## 次やること
- GPU, HifiGANの問題を調査
- データをとって集める
