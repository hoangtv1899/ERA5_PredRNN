import os.path
import datetime
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from core.utils import preprocess, metrics
import lpips
import torch
import wandb

from scipy import ndimage

def center_enhance(img, min_distance = 100, sigma=4, radii=np.arange(0, 20, 2),find_max=True,enhance=True,multiply=2):
    if enhance:
        filter_blurred = ndimage.gaussian_filter(img,1)
        res_img = img + 30*(img - filter_blurred)
    else:
        res_img = ndimage.gaussian_filter(img,3)
    return res_img

loss_fn_alex = lpips.LPIPS(net='alex')


def train(model, ims, real_input_flag, extra_var, configs, itr):
    torch.cuda.empty_cache()
    model.train(ims, real_input_flag, extra_var)

def validate(model, test_input_handle, extra_var, configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'testing...')
    if extra_var.all() is not None:
        extra_var = torch.FloatTensor(extra_var).to(configs.device)
    
    output_length = configs.total_length - configs.input_length

    if configs.reverse_scheduled_sampling == 1:
        mask_input = 1
    else:
        mask_input = configs.input_length

    real_input_flag = torch.zeros((configs.test_batch_size, configs.total_length-mask_input-1, 1, 1, 1))
    # print(f"real_input_flag: {real_input_flag.shape}")
    if configs.reverse_scheduled_sampling == 1:
        real_input_flag[:, :configs.input_length - 1, :, :] = 1.0
    
    real_input_flag = torch.FloatTensor(real_input_flag).to(configs.device)
    test_ims = test_input_handle.get_batch()
    test_ims = torch.FloatTensor(test_ims).to(configs.device)

    torch.cuda.empty_cache()
    avg_mse, loss = model.test(test_ims, real_input_flag, extra_var)
        
    # for i in range(configs.total_length-1):
    #     mse_i = np.mean((img_out[:,i]-test_ims[:,i+1])**2*configs.area_weight)
    #     img_mse.append(mse_i)
    #     print(i, mse_i)
    print(f"{configs.save_file}, loss: {loss.mean()}, avg_mse: {avg_mse}")
    if configs.upload_run:
        wandb.log({"Test mse": float(avg_mse)})
    return avg_mse


def test(model, test_input_handle, real_input_flag, extra_var, configs, save_data_name):
    '''
    When using this function, you must set configs.concurent_step > 1
    '''
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'testing...')
    # reverse schedule sampling
    
    if extra_var.all() is not None:
        extra_var = torch.FloatTensor(extra_var).to(configs.device)
    
    output_length = configs.total_length - configs.input_length

    if configs.save_test_result:
        total_batch = (test_input_handle.total())//configs.test_batch_size*configs.test_batch_size
        shape = (total_batch, configs.concurent_step*output_length+configs.input_length,
                configs.img_channel, configs.img_height, configs.img_width)
        perd_data_name = '_'.join([save_data_name, configs.save_file, 'pred.dat'])
        if os.path.isfile(perd_data_name):
            return
        true_data_name = '_'.join([save_data_name, configs.save_file, 'true.dat'])
        pred_data_array = np.memmap(configs.gen_data_dir+perd_data_name, dtype='float32', mode='w+', shape=shape)
        true_data_array = np.memmap(configs.gen_data_dir+true_data_name, dtype='float32', mode='w+', shape=shape)
    
    i = 0
    acc_mse = 0
    cur_pos = 0

    while not test_input_handle.no_batch_left():
        test_ims = test_input_handle.get_batch()
        B = test_ims.shape[0]
        for bi in range(B):
            if configs.save_test_result:
                true_data_array[cur_pos+bi,:] = test_ims[bi,...]
            
            test_imsi = test_ims[bi,...][np.newaxis,...]
            print(f"{i} + {bi}, test_ims shape: {test_imsi.shape}")
            test_imsi = torch.FloatTensor(test_imsi).to(configs.device)
            img_out, loss = model.test(test_imsi, real_input_flag[0:1], extra_var)

            if isinstance(img_out, np.ndarray):
                img_out = torch.from_numpy(img_out).to(configs.device)

            img_out = img_out.detach()

            if configs.save_test_result:
                pred_data_array[cur_pos+bi,:] = img_out.cpu().numpy()


            print(f"test_ims shape: {test_imsi.shape}, img_out shape: {img_out.shape}")
            #avg_mse = torch.mean((img_out[:,-output_length:]-test_imsi[:,-output_length:])**2*configs.area_weight).cpu().numpy()
            #acc_mse += avg_mse
            #print(f"{configs.save_file}, loss: {loss.mean()}, avg_mse: {avg_mse}")
            del img_out
            del test_imsi
            torch.cuda.empty_cache()
        cur_pos += B
        i += 1
        test_input_handle.next()
    #print(f"The avg mse is {acc_mse/i}")
    if configs.save_test_result:
        true_data_array.flush()
        pred_data_array.flush()
