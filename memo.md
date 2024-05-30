## 環境初期化
```
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
```
