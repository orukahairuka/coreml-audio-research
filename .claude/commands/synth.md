録音→合成→再生を実行します。

引数 $ARGUMENTS が指定されていればそのファイルを、なければ input_sample.wav を入力として使います。

以下の手順で実行してください:

1. 入力ファイルの存在確認（PronounSE/ ディレクトリ内で探す）
2. `cd PronounSE && venv/bin/python synthesis.py <入力ファイル>` を実行
3. 合成が成功したら、PronounSE/result/ 内の最新 wav ファイルを `afplay` で再生
4. 結果のサマリーを表示（入力ファイル名、出力ファイル名）
