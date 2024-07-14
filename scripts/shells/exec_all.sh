#!/bin/bash
set -eux

export TORCH_CUDA_ARCH_LIST="8.6"

cd $(dirname $0)/../../
dataset_dir=$1

cuda_id=0
date -ud @$SECONDS "+%T"

# Train model
bash scripts/shells/train.sh $dataset_dir $cuda_id
date -ud @$SECONDS "+%T"

# output/street-gaussians-ns/street-gaussians-ns/ 以下の最新日付のディレクトリを取得
latest_output_dir=$(ls -td $dataset_dir/../output/street-gaussians-ns/street-gaussians-ns/* | head -n 1)

# Render model
bash scripts/shells/render.sh $latest_output_dir/config.yml $cuda_id
date -ud @$SECONDS "+%T"

# Eval model
bash scripts/shells/eval.sh $latest_output_dir/config.yml $cuda_id
date -ud @$SECONDS "+%T"

# concat
python3 scripts/pythons/concat_images.py \
  ${latest_output_dir}/renders/all/gt-rgb \
  ${latest_output_dir}/renders/all/rgb \
  ${latest_output_dir}/renders/all/concat \
  --text1 "GT" \
  --text2 "Street-Gaussigns-NS" \
  --ext "jpg"
date -ud @$SECONDS "+%T"
