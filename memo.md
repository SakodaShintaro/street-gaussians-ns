# 手順メモ

## 環境初期化

```bash
python3 -m venv .env
source .env/bin/activate
pip3 install torch torchvision
git clone https://github.com/SakodaShintaro/street-gaussians-ns
git submodule update --init --recursive
pip3 install rawpy==0.21.0
pip3 install numpy==1.24.3 pandas==2.2.1
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
sudo apt install -y ffmpeg

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

## Dockerを使って環境構築する場合

nerf-studioのドキュメントを参考にする

<https://docs.nerf.studio/quickstart/installation.html#use-docker-image>

<https://hub.docker.com/r/dromni/nerfstudio>

を使う。

```bash
git clone https://github.com/SakodaShintaro/street-gaussians-ns
cd street-gaussians-ns/

# イメージをDockerfileから作る
docker build -t my-pytorch-image .

docker run --gpus all \
            -v $HOME/data:/home/user/data/ \
            -v $HOME/.cache/:/home/user/.cache/ \
            -v $(readlink -f ./street-gaussians-ns/):/home/user/street-gaussians-ns
            -v /media:/media \
            -p 7007:7007 \
            -it \
            --ipc=host \
            --privileged \
            my-pytorch-image \
            /bin/bash

sudo apt update
sudo apt install -y ffmpeg libgl1-mesa-glx libglib2.0-0 libapparmor1 libglapi-mesa libglx-mesa0 libgl1-mesa-dri libgbm1

git clone https://github.com/SakodaShintaro/street-gaussians-ns
cd street-gaussians-ns/

git submodule update --init --recursive
pip3 install "git+https://github.com/facebookresearch/pytorch3d.git"
pip3 install -e .
pip3 install dependencies/nvdiffrast/
cd dependencies/detectron2
pip3 install -e .
cd ../Mask2Former
pip3 install -r requirements.txt

# sh make.sh はそのままだと権限の問題で上手くいかない
# make.shスクリプトの末尾に--userをつけてから実行する
# 例) python3 setup.py build install --user
vim make.sh
sh make.sh

sudo apt install tzdata

cd ~/street-gaussians-ns/
wget -P dependencies/Mask2Former/models/ https://dl.fbaipublicfiles.com/maskformer/mask2former/mapillary_vistas/semantic/maskformer2_swin_large_IN21k_384_bs16_300k/model_final_90ee2d.pkl
```

### 実行

```bash
export PATH=$PATH:~/.local/bin
pip3 install nerfstudio==1.1.4
bash ./scripts/shells/train.sh ~/data/rosbag/20241221_for_3dgs/train_data/
```

### ros2 (humble)

https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html

```
locale  # check for UTF-8

sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

locale  # verify settings

sudo apt install software-properties-common
sudo add-apt-repository universe

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update

sudo apt install ros-humble-ros-base
source /opt/ros/humble/setup.bash

pip3 install opencv-python cv_bridge
```
