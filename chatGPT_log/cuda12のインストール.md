狙いは “`libcudart.so.12`（CUDA Runtime）を APT で入れる” ことです。環境によってパッケージ名が2通りあります。

## 1) Ubuntu 公式にある場合（新しめのUbuntu）

Ubuntuのリポジトリに `libcudart12` がある世代ならこれでOK：

```bash
sudo apt update
sudo apt install libcudart12
```

Ubuntuの Launchpad では `libcudart12` が「NVIDIA CUDA Runtime Library」として提供されています。([Launchpad][1])

## 2) NVIDIA の CUDA APT リポジトリを使う場合（推奨・細かく版指定可）

まず NVIDIA のキーリングを追加（例：Ubuntu 24.04/22.04 共通手順）：

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
# 22.04なら URL の ubuntu2404 を ubuntu2204 に変えてください
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
```

この後、**ランタイムだけ**欲しいなら版を指定して：

```bash
# 例：CUDA 12.4 のランタイム一式（cudart を含む）
sudo apt install cuda-runtime-12-4
```

もしくは **cudart 単体**をピンポイントで：

```bash
# cudart 単体（CUDA 12.4）
sudo apt install cuda-cudart-12-4
```

NVIDIA 公式の Dockerfile 等でも `cuda-cudart-12-4` というパッケージ名でインストールしています（NVIDIA 配布の APT/RPM で共通の命名）。([about.gitlab.com][2])
NVIDIA の CUDA Linux インストールガイドも APT レポ経由の導入を案内しています。([NVIDIA Docs][3])

---

## 入ったか確認

```bash
# どの libcudart が見えているか
ldconfig -p | grep libcudart

# シンボリックリンクの実体
readlink -f /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12
```

## うまく見つからないとき

```bash
apt-cache policy libcudart12 cuda-runtime-12-4 cuda-cudart-12-4
```

で、どのパッケージ名が利用可能かを確認してください（ディストリや追加済みレポにより異なります）。

* 公式Ubuntuだけ → `libcudart12`
* NVIDIA レポ追加後 → `cuda-cudart-12-*` / `cuda-runtime-12-*` / `cuda-toolkit-12-*` が候補になります。([developer.download.nvidia.com][4])

> 参考：`libcudart.so.12` は CUDA 12 系のランタイムで、ファイル名に 12.x の完全版番号が付きます（例 `libcudart.so.12.4.127`）。`libcudart.so` → `...so.12` → `...so.12.4.127` の順にシンボリックリンクされています。これを見れば入っている“具体的なビルド版”が分かります。 ([NVIDIA Developer Forums][5])

[1]: https://answers.launchpad.net/ubuntu/plucky/%2Bpackage/libcudart12?utm_source=chatgpt.com "libcudart12 : Plucky (25.04) : Ubuntu - Launchpad Answers"
[2]: https://gitlab.com/nvidia/container-images/cuda/blob/master/dist/12.4.1/ubuntu2204/base/Dockerfile?utm_source=chatgpt.com "dist/12.4.1/ubuntu2204/base/Dockerfile · master · nvidia / ..."
[3]: https://docs.nvidia.com/cuda/archive/12.4.0/cuda-installation-guide-linux/?utm_source=chatgpt.com "1. Introduction — Installation Guide for Linux 12.4 ..."
[4]: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/Packages "developer.download.nvidia.com"
[5]: https://forums.developer.nvidia.com/t/no-libcudart-so-12-with-cuda-toolkit/253270?utm_source=chatgpt.com "No libcudart.so.12 with cuda-toolkit"
