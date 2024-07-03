#!/bin/bash
set -eux

cd $(dirname $0)/../../

python -m pip install --upgrade pip
pip3 install rawpy==0.21.0 numpy==1.24.3 pandas==2.2.1 ninja
pip3 install "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
pip3 install nerfstudio
pip3 install "git+https://github.com/facebookresearch/pytorch3d.git"
pip3 install -e .
pip3 install -e dependencies/nvdiffrast
pip3 install -e dependencies/detectron2
pip3 install -r dependencies/Mask2Former/requirements.txt
