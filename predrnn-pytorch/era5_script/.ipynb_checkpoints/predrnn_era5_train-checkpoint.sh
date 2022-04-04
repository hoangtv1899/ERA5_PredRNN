export CUDA_VISIBLE_DEVICES=0
cd ..
python -u run.py \
    --is_training 1 \
    --device cuda \
    --dataset_name mnist \
    --train_data_paths /scratch/network/hvtran/era5/era5_train.npz \
    --valid_data_paths /scratch/network/hvtran/era5/era5_valid.npz \
    --save_dir /scratch/network/hvtran/era5/predrnn-pytorch/checkpoints/era5_predrnn \
    --gen_frm_dir /scratch/network/hvtran/era5/predrnn-pytorch/checkpoints/era5_predrnn \
    --model_name predrnn_v2 \
    --reverse_input 1 \
    --img_width 720 \
    --img_channel 5 \
    --input_length 6 \
    --total_length 12 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 40 \
    --layer_norm 1 \
    --scheduled_sampling 1 \
    --sampling_stop_iter 500 \
    --sampling_start_value 1.0 \
    --sampling_changing_rate 0.00002 \
    --lr 0.01 \
    --batch_size 1 \
    --max_iterations 2000 \
    --display_interval 10 \
    --test_interval 50000 \
    --snapshot_interval 500 \
    --pretrained_model /home/hvtran/model.ckpt-2000

cp /scratch/network/hvtran/era5/predrnn-pytorch/checkpoints/era5_predrnn/model.ckpt-2000 /home/hvtran/