export CUDA_VISIBLE_DEVICES=0,1,2
cd ..
python -u run1.py \
    --is_training 0 \
    --concurent_step 14 \
    --device cuda \
    --dataset_name mnist \
    --train_data_paths /work/08589/hvtran/ls6/ERA5_PredRNN-main/era5_train_1024002012_3_24hr.npz \
    --valid_data_paths /work/08589/hvtran/ls6/ERA5_PredRNN-main/era5_train_1024002012_3_24hr.npz \
    --save_dir /scratch/08589/hvtran/predrnn-pytorch/checkpoints/2012_predrnn_test_sea_press \
    --gen_frm_dir /scratch/08589/hvtran/predrnn-pytorch/checkpoints/2012_predrnn_test_sea_press \
    --model_name predrnn_v2 \
    --reverse_input 1 \
    --img_height 720 \
    --img_width 1440 \
    --use_weight 0 \
    --layer_weight 10,10,10,10,20,20 \
    --img_channel 1 \
    --img_layers 1,0,2 \
    --input_length 24 \
    --total_length 48 \
    --num_hidden 256,256,256,256  \
    --skip_time 1 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 15 \
    --layer_norm 1 \
    --decouple_beta 0.05 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 25000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2500 \
    --lr 0.05 \
    --batch_size 1 \
    --max_iterations 4000 \
    --display_interval 10 \
    --test_interval 200 \
    --snapshot_interval 200 \
    --conv_on_input 0 \
    --res_on_conv 0 \
    --is_WV 0 \
    --center_enhance True \
    --layer_need_enhance 0 \
    --find_max False \
    --multiply 1.1 \
    --pretrained_model /work/08589/hvtran/ls6/ERA5_PredRNN-main/model.ckpt-best-sea_press

#cp /scratch/network/hvtran/era5/predrnn-pytorch/checkpoints/era5_predrnn/model.ckpt-1000 /home/hvtran/
# /work/08589/hvtran/ls6/ERA5_PredRNN-main/era5_train_0921002022_3_24hr.npz 
