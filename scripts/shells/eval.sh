config_path=$1
cuda_id=$2

CUDA_VISIBLE_DEVICES=$cuda_id sgn-eval \
    --load-config $config_path \
    --render_output_path $(dirname $config_path)/eval_render_output
