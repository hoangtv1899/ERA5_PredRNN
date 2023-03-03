export CUDA_VISIBLE_DEVICES=0,1,2
cd ..
python -u run1.py \
    --is_training 1 \
    --device cuda \
    --dataset_name mnist \
    --train_data_paths /work/08589/hvtran/ls6/ERA5_PredRNN-main/era5_train_1024002012_3_24hr.npz,/work/08589/hvtran/ls6/ERA5_PredRNN-main/era5_train_1001002015_3_24hr.npz,/work/08589/hvtran/ls6/ERA5_PredRNN-main/era5_train_1001002016_3_24hr.npz,/work/08589/hvtran/ls6/ERA5_PredRNN-main/era5_train_0827002021_3_24hr.npz \
    --valid_data_paths /work/08589/hvtran/ls6/ERA5_PredRNN-main/era5_train_0921002022_3_24hr.npz \
    --save_dir /work/08589/hvtran/ls6/ERA5_PredRNN-main/predrnn-pytorch/checkpoints/era5_predrnn_precip \
    --gen_frm_dir /work/08589/hvtran/ls6/ERA5_PredRNN-main/predrnn-pytorch/checkpoints/era5_predrnn_precip \
    --model_name predrnn_v2 \
    --reverse_input 1 \
    --is_WV 1 \
    --center_enhance False \
    --layer_need_enhance 0 \
    --find_max False \
    --multiply 2 \
    --img_height 720 \
    --img_width 1440 \
    --use_weight 0 \
    --layer_weight 20 \
    --img_channel 1 \
    --img_layers 2,1,0 \
    --input_length 24 \
    --total_length 48 \
    --num_hidden 460,460,460,460,460,460 \
    --skip_time 1 \
    --wavelet db1 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 15 \
    --layer_norm 1 \
    --decouple_beta 0.05 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 25000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2500 \
    --lr 1e-4 \
    --batch_size 1 \
    --max_iterations 10000 \
    --display_interval 50 \
    --test_interval 1000000 \
    --snapshot_interval 2000 \
    --conv_on_input 0 \
    --res_on_conv 0 \
    --curr_best_loss 0.011 \
    --pretrained_model /work/08589/hvtran/ls6/ERA5_PredRNN-main/model.ckpt-best-precip

#cp /scratch/network/hvtran/era5/predrnn-pytorch/checkpoints/era5_predrnn/model.ckpt-1000 /home/hvtran/
#,/work/08589/hvtran/ls6/ERA5_PredRNN-main/era5_train_1001002015_3_24hr.npz,/work/08589/hvtran/ls6/ERA5_PredRNN-main/era5_train_1001002016_3_24hr.npz,/work/08589/hvtran/ls6/ERA5_PredRNN-main/era5_train_0827002021_3_24hr.npz
