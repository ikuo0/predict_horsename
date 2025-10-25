55秒はさすがに長いです。**devcontainer でも毎回 50秒超は“何かがおかしい”部類**。典型的には「ファイルシステムの遅さ（バインドマウント）」＋「TFが大量の.soやカーネルを探して登録する」二重パンチで起きます。切り分けと対処を一気にまとめます。

# まず切り分け（最短で原因特定）

1. **バインドマウントの影響を除外**
   同じコンテナで「ワークスペースを**マウントしない**」状態のベースラインを測ります。

   ```bash
   docker run --rm -it tensorflow/tensorflow:latest \
     python -c 'import time; t=time.time(); import tensorflow as tf; print("import:", round(time.time()-t,2),"s")'
   ```

   * これが **数秒〜十数秒** なら、あなたの devcontainer の**マウント先（Windows側/外部FS）**がボトルネックの可能性大。
   * ここでも 50秒級なら、イメージやCPU拡張、動的リンカ周りの問題を疑います。

2. **`importtime` でどこに時間を食ってるか可視化**

   ```bash
   python -X importtime -c "import tensorflow" 2> tf.importlog
   python -m pip install tuna >/dev/null 2>&1 || true
   python -m tuna tf.importlog  # ブラウザでツリー表示（どのモジュールが重いか）
   ```

   “oneDNN/PluggableDevice/keras/backend”あたりや、`.so` ロード部分に時間が集中していないか確認。

3. **ファイル I/O が遅いか確認（マウント/ウイルス対策）**

   * devcontainer が **Windows/WSL のバインドマウント**だと、**数十万ファイルの stat/open** が極端に遅くなることがあります（特に HDD）。
   * `strace` でファイル系が支配的か確認：

     ```bash
     strace -f -T -e trace=file python -c "import tensorflow" 2>&1 | tail
     ```

# すぐ効く対策（実効性が高い順）

* **(A) マウント先を速くする**

  * ワークスペースを **WSL2 の ext4（Linux側）** に置く（Windows側のパスをそのままマウントしない）。
  * あるいは `devcontainer.json` でソースは bind mount のままでも、**`site-packages` だけはコンテナ内に置く**（＝イメージ内に `pip install` して、Python が多数の .py/.so を**ローカル ext4**から読むようにする）。
  * 最終手段：実験用は **Docker volume** にクローンして動かす。

* **(B) CPU限定＋初期化軽量化**（あなたが既にセットしたもの＋追加）

  ```python
  import os
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # GPU検出スキップ（最重要）
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # ログ抑制
  os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
  os.environ["TF_NUM_INTEROP_THREADS"] = "1"
  os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # oneDNN大量登録のコストを下げられる環境がある
  # （↑ CPU演算は遅くなる可能性。起動だけ速くしたい“試験用途”ならアリ）
  import tensorflow as tf
  ```

  * さらに **CPU版のWheel** を使えるTFバージョンなら、`pip install tensorflow-cpu==<ver>`（または最近系は `tensorflow` がCPU版）にして**GPU関連.so探索をなくす**。

* **(C) 遅延 import の徹底**

  * TF を使う関数の**内部で** `import tensorflow as tf` する（他モジュールのトップレベルに TF import を置かない）。
  * 依存ツリーのどこかで `from tensorflow import keras` などをトップレベルに書いていると、結局早期に import されます。依存を切り分け。

* **(D) 余計な衝突を避ける**

  * **PyTorch等を先に import** すると OpenMP/MKL スレッド初期化が重複・競合することがあります。TFを単独で計測してください。
  * `PYTHONPATH` に巨大な検索パスを積んでいないか、`.pth` の副作用で遅くなっていないか確認。

# 追加のチェックポイント

* **バージョン差**：同じ devcontainer でも `tensorflow==2.10` と `2.17` 以降では初期化の重さや oneDNN の扱いが変わります。**別タグの公式イメージ**でベースラインを取ると早い。
* **動的リンカ**：`LD_DEBUG=libs` を付けて `.so` 解決に異常な待ちがないか確認。
* **アンチウイルス**：Windows 側に置いている場合、リアルタイムスキャンで .py/.so のアクセスが遅くなることがあります。**除外設定**を試す価値あり。

# 実用ショートカット

* **“起動時間だけ”短縮したい**最小例：

  ```bash
  # devcontainer に入ってから（まず環境のせいかどうかのベースライン）
  python - <<'PY'
  import os, time
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
  os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
  os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
  os.environ["TF_NUM_INTEROP_THREADS"] = "1"
  t=time.time()
  import tensorflow as tf
  print("import:", round(time.time()-t,2), "s", "ver:", tf.__version__)
  PY
  ```

  これで**数秒〜十数秒**に落ちるなら OK、**まだ数十秒**ならマウント/FS が主犯。

---

## 結論

* devcontainer でも **毎回 55秒は異常域**。多くは **Windows側バインドマウント + HDD** で Python が大量ファイルを触る時に起きます。
* **WSL/Linux側に置く、`site-packages` をイメージ内に持つ、GPU検出を切る、oneDNN最適化を一時停止** の組み合わせで、体感を大きく改善できます。
* まずは「**バインドマウントなしベースライン**」を計測して、問題の主犯を確定させるのが最短です。
