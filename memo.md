# 手順メモ

## 環境初期化

```bash
python3 -m venv .env
source .env/bin/activate
pip3 install torch torchvision
pip3 install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip3 install nerfstudio
pip3 install "git+https://github.com/facebookresearch/pytorch3d.git"
pip3 install -e .
pip3 install dependencies/nvdiffrast/
cd dependencies/detectron2
pip3 install -e .
cd ../Mask2Former
pip3 install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh

pip3 install --upgrade pip
pip3 install waymo-open-dataset-tf-2-12-0
```

## colmap

aptで入れたもの
`COLMAP 3.7 (Commit Unknown on Unknown without CUDA)` では `colmap model_comparer` というコマンドが存在しなかった。

ソースコードから3.9.1をビルドする。
<https://colmap.github.io/install.html>

```bash
sudo apt-get remove --purge colmap
git clone https://github.com/colmap/colmap.git
cd colmap
git checkout tags/3.9.1
mkdir build
cd build
<CMakeLists.txtに set(CMAKE_CUDA_ARCHITECTURES "native")を追加>
cmake .. -GNinja
ninja
sudo ninja install
```

## 学習実行

```bash
python3 ./scripts/pythons/extract_waymo.py --waymo_root ./data/ --out_root ./data/extracted
bash scripts/shells/data_process.sh ./data/extracted
bash scripts/shells/train.sh ./data/extracted 0
```
