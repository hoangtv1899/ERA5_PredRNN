export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ..
base_path="/pscratch/sd/h/hvtran/ERA5/dataset/"
train_data_paths=""
for file in /pscratch/sd/h/hvtran/ERA5/dataset_1hr/era5_train*.npz; do
    train_data_paths+="${file},"
done
load_path="/global/u2/h/hvtran/haoli/PredRNN_checkpoints/WV_1_Mass_0_E_0_1hr/"

# Remove the trailing comma
train_data_paths=${train_data_paths%,}

python -u run1.py \
    --is_training 1 \
    --device cuda:0 \
    --dataset_name mnist \
    --train_data_paths ${train_data_paths} \
    --valid_data_paths /pscratch/sd/h/hvtran/ERA5/dataset_1hr/era5_train_1979_sel_0_6_24hr.npz \
    --save_dir /global/u2/h/hvtran/haoli/PredRNN_checkpoints/WV_1_Mass_0_E_0_1hr/ \
    --gen_frm_dir /global/u2/h/hvtran/haoli/PredRNN_checkpoints/WV_1_Mass_0_E_0_1hr/ \
    --model_name predrnn_v2 \
    --reverse_input 0 \
    --test_batch_size 1\
    --add_geopential 0 \
    --gpu_num 4 \
    --add_land 0 \
    --add_latitude 0 \
    --is_WV 1 \
    --press_constraint 0 \
    --press_layer 2 \
    --dry_static_constraint 0 \
    --temp_layer 4 \
    --global_dem_file /global/u2/h/hvtran/haoli/ERA5_PredRNN/predrnn-pytorch/core/utils/elev.0.25-deg.nc \
    --center_enhance 0 \
    --patch_size 30 \
    --weighted_loss 1 \
    --upload_run 0 \
    --layer_need_enhance 1 \
    --find_max False \
    --multiply 2 \
    --img_height 720 \
    --img_width 1440 \
    --use_weight 0 \
    --layer_weight 20 \
    --img_channel 5 \
    --img_layers 0,1,2,3,4 \
    --input_length 12 \
    --total_length 24 \
    --time_step 1hr \
    --num_hidden 800,800,800,800 \
    --skip_time 1 \
    --wavelet db1 \
    --filter_size 5 \
    --stride 1 \
    --layer_norm 1 \
    --decouple_beta 0.05 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 25000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2500 \
    --lr 5e-4 \
    --batch_size 4 \
    --max_iterations 5000 \
    --display_interval 1000 \
    --test_interval 20000 \
    --snapshot_interval 1000 \
    --conv_on_input 0 \
    --res_on_conv 0 \
    --curr_best_mse 0.025 \
    --save_best_name wv1_pc0 \
    --pretrained_model ${load_path} \
    --pretrained_model_name model_final.ckpt \
