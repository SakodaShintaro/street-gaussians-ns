#!/bin/bash
set -eux

config_path=$1
cuda_id=$2

CUDA_VISIBLE_DEVICES=$cuda_id sgn-render \
    --load-config $config_path \
    --downscale_factor 2
