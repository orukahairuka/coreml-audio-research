PyTorch モデルを CoreML (.mlpackage) に変換します。

$ARGUMENTS で精度を指定できます（float16 / float32 / int8）。未指定時は float16 です。

以下の2つのスクリプトを順に実行してください:

1. `PronounSE/venv/bin/python scripts/convert_transformer.py --precision <精度>` — Encoder と Decoder を変換
2. `PronounSE/venv/bin/python scripts/convert_hifigan.py --precision <精度>` — HiFi-GAN Generator を変換

実行後、プロジェクトルートに生成された `.mlpackage` ファイルを一覧表示してください。
