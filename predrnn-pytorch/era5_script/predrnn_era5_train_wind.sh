export CUDA_VISIBLE_DEVICES=0
cd ..
python -u run1.py \
    --is_training 1 \
    --device cuda \
    --dataset_name mnist \
    --train_data_paths /work/08589/hvtran/ls6/ERA5_PredRNN-main/era5_train_5_12hr.npz \
    --valid_data_paths /work/08589/hvtran/ls6/ERA5_PredRNN-main/era5_train_5_12hr.npz \
    --save_dir /work/08589/hvtran/ls6/ERA5_PredRNN-main/predrnn-pytorch/checkpoints/era5_predrnn \
    --gen_frm_dir /work/08589/hvtran/ls6/ERA5_PredRNN-main/predrnn-pytorch/checkpoints/era5_predrnn \
    --model_name predrnn_v2 \
    --reverse_input 1 \
    --img_height 720 \
    --img_width 1440 \
    --use_weight 1 \
    --layer_weight 20,20,10,10,10 \
    --img_channel 5 \
    --input_length 12 \
    --total_length 24 \
    --num_hidden 512,512,512,512 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 40 \
    --layer_norm 1 \
    --decouple_beta 0.05 \
    --reverse_scheduled_sampling 1 \
    --r_sampling_step_1 25000 \
    --r_sampling_step_2 50000 \
    --r_exp_alpha 2500 \
    --lr 2.5e-4 \
    --batch_size 1 \
    --max_iterations 2000 \
    --display_interval 10 \
    --test_interval 1000000 \
    --snapshot_interval 100 \
    --conv_on_input 0 \
    --res_on_conv 0 \
    --pretrained_model /work/08589/hvtran/ls6/ERA5_PredRNN-main/model.ckpt-1000

#cp /scratch/network/hvtran/era5/predrnn-pytorch/checkpoints/era5_predrnn/model.ckpt-1000 /home/hvtran/
