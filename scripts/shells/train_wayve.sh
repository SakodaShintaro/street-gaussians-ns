data_root=$1
cuda_id=0

mkdir -p output/

CUDA_VISIBLE_DEVICES=$cuda_id  sgn-train street-gaussians-ns \
    --experiment_name street-gaussians-ns \
    --output_dir $data_root/../output_wayve/ \
    --vis viewer+tensorboard \
    --viewer.quit_on_train_completion True \
    colmap-data-parser-config \
    --data $data_root \
    --colmap_path colmap_sparse/rig_txt \
    --load_3D_points True \
    --max_2D_matches_per_3D_point 0 \
    --undistort True \
    --init_points_filename points3D.txt
