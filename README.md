# 競走馬の名前を学習して、新しい名前を生成するモデルを作成するプロジェクト

## 手順
scraping netkeibaからデータのダウンロード
parse HTMLをパースしてテーブルデータをJSONで保存
summary 登場する文字を集計（カタカナ文字のみ）
futurize Xデータ、Yデータ に特徴化、正解ラベルの作成を実施
scaling_feature
PCA 8~64 まで圧縮して再現度を見て使う奴を決める
  PCAは不要、圧縮率が悪く、圧縮するメリットが無い
model_trainer でモデル学習
  学習済みモデルを保存
  JavaScript で使えるように tensorflowjs で変換
model_predict で学習済みモデルの試験

## 除外コード
scraping は netkeiba に迷惑かかりそうなのでコード非公開
その他競走馬名や学習データはサイズや数が大きすぎるため非公開

# 以下メモ

# tensorflowについて
https://www.tensorflow.org/js/tutorials?hl=ja


# netkeiba HTMLについて

```
<meta http-equiv="content-type" content="text/html; charset=euc-jp" />
```

EUC-JP でエンコードされているため、UTF-8 に変換してからパースする必要がある。

```
iconv -f euc-jp -t utf-8 /workspaces/pj0005_horse_name/scraping/download_data/page_00001.html -o /workspaces/pj0005_horse_name/scraping/download_data/page_00001.utf8.html

# 入力: EUC-JP → 出力: UTF-8、別名で保存
nkf --ic=euc-jp --oc=utf-8 scraping/download_data/page_00001.html > scraping/download_data/page_00001.utf8.html

```


# pip
```
pip install beautifulsoup4
pip install numpy
pip install sqlite-vec
pip install sqlite-vss
pip install sqlite-utils
sqlite-utils install sqlite-utils-sqlite-vec
conda install -c conda-forge sqlite-fts4
```

# コマンドメモ

```

# 名前順ソートでファイル列挙
ls -1

# ファイルの数数える
find ./parse/parse_data/ -type f | wc -l
find ./scraping/download_data -type f | wc -l

# keras の古いバージョンを使う、メタ情報が埋まりやすくなる
export TF_USE_LEGACY_KERAS=1

```


# GPU 認識対応

devcontainer、Dockerなどの問題ではなく
CUDA ドライバのバージョンが古いと認識されないことが原因だった
list_physical_devices("GPU") で GPU 列挙時に 0 個の時は "XLAGPU" を指定すると認識されることもあった
最終的には "GPU" で認識できた、CUDA ドライバを最新にしたら認識された

# tensorflowjs の使い方
python.tensorflow 側で `model = keras.Model` をトレーニング後に `model.export(export_dir, "tf_saved_model")` で保存し、
次のコマンドでJS用の変換を実行

```
        tensorflowjs_converter \
        --input_format=tf_saved_model \
        --saved_model_tags=serve \
        model_trainer/model_trainer_out/saved_model \
        model_trainer/model_trainer_out/saved_model_out
```

tf.loadGraphModel で読み込みという手順で上手く動作した。

.h5, .keras 形式からの変換 → tf.loadLayersModel では上手くいかなかった

