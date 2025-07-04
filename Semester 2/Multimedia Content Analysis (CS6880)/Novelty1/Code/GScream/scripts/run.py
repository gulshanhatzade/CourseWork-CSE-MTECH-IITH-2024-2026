import os
import sys
from tqdm import tqdm


gpuid = '0'
task_name = f"gscream"
refs_root = './refs'

voxel_size=0.005
update_init_factor=16
os.system("ulimit -n 4096")

for nn in ["1", "3", "10", "trash"]:
    ref_image_path = os.path.join(refs_root, [i for i in sorted(os.listdir(refs_root)) if i.startswith(nn+'_out') and i.endswith('png')][0])
    print(ref_image_path)
    assert os.path.exists(ref_image_path), ref_image_path
    ref_depth_path = os.path.join(refs_root, [i for i in sorted(os.listdir(refs_root)) if i.startswith(nn+'_out') and i.endswith('npy')][0])
    print(ref_depth_path)
    assert os.path.exists(ref_depth_path), ref_depth_path

    image_root = f"images"

    cmd = f"python -u train.py \
        -s /kaggle/input/spinnerf-dataset-processed/spinnerf_dataset_processed/{nn} \
        --eval \
        --lod 0 \
        --gpu 0 \
        --voxel_size {voxel_size} \
        --update_init_factor {update_init_factor} \
        --iterations 30_000 \
        --port 10001 \
        -m outputs/spinnerf_dataset/{nn}/{task_name}/ \
        --is_spin \
        --images {image_root} \
        --specified_ply_path /kaggle/input/spinnerf-dataset-processed/spinnerf_dataset_processed/{nn}/sparse/0/points3D.ply \
        --ref_image_path {ref_image_path} \
        --ref_depth_path {ref_depth_path} \
        --load_mask \
        --load_depth \
        --load_norm \
        --lpips_lr 0 \
        --lpips_b 20 \
        --perceptual_lr 0 \
        --perceptual_b 2 \
        --refer_rgb_lr 1.0 \
        --refer_rgb_lr_fg 20.0 \
        --other_rgb_lr 1.0 \
        --other_rgb_lr_fg 0.0 \
        --refer_depth_lr 1 \
        --refer_depth_lr_fg 100 \
        --refer_depth_lr_smooth 1 \
        --other_depth_lr 0.1 \
        --other_depth_lr_smooth 0.1 \
        --refer_normal_lr 0 \
        --other_normal_lr 0 \
        --refer_opacity_lr 0 \
        --other_opacity_lr 0 \
        --uncertainty_lr 0 \
        --vgg_lr 0 \
        --discriminator_lr 0 \
        --adv_lr 0 \
        --crossattn_lr_init 0.002 \
        --crossattn_lr_final 0.00002 \
        --crossattn_lr_delay_mult 0.01 \
        --crossattn_lr_max_steps 30000 \
        --enable_crossattn_refview 1.0 \
        --enable_crossattn_otherview 1.0 \
        --attn_head_num 8 \
        --attn_head_dim 64 \
        --enable_enlarge_samping 0.0 \
        --sampling_2D_enlarge_ratio 2.0 \
        --enable_edge_samping 1.0 \
        --sampling_2D_small_ratio 0.6 \
        --enable_pe 0.0 \
        --crossattn_feat_update_ema 0.03 \
        --start_crossattn_from 15000 \
        --perceptual_loss_weight 0.1 \
        --gradient_domain_loss_weight 0.5 \
        --boundary_emphasis \
        > log_training/log_train_spin-{nn}_{task_name}.txt"
    os.system(cmd)
    # print(cmd) 

