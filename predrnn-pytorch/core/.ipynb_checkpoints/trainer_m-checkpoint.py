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


def train(model, ims, real_input_flag, configs, itr):
    cost = model.train(ims, real_input_flag)
    if configs.reverse_input:
        ims_rev = np.flip(ims, axis=1).copy()
        cost += model.train(ims_rev, real_input_flag)
        cost = cost / 2
    
    if itr % configs.display_interval == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
        print('training loss: ' + str(cost))


def test(model, test_input_handle, configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'testing...')
    # reverse schedule sampling
    if configs.reverse_scheduled_sampling == 1:
        mask_input = 1
    else:
        mask_input = configs.input_length

    real_input_flag = torch.zeros((configs.test_batch_size, configs.total_length-mask_input-1, 1, 1, 1))
    # print(f"real_input_flag: {real_input_flag.shape}")
    if configs.reverse_scheduled_sampling == 1:
        real_input_flag[:, :configs.input_length - 1, :, :] = 1.0
    test_ims = test_input_handle.get_batch()
    real_input_flag = torch.FloatTensor(real_input_flag).to(configs.device)
    test_ims = torch.FloatTensor(test_ims).to(configs.device)
    output_length = configs.total_length - configs.input_length
    torch.cuda.empty_cache()
    img_out, loss, loss_pred, decouple_loss = model.test(test_ims, real_input_flag)
    img_out = img_out.detach()
    # print(f"test_ims shape: {test_ims.shape}, img_out shape: {img_out.shape}")
        
    img_mse = []
    # for i in range(configs.total_length-1):
    #     mse_i = np.mean((img_out[:,i]-test_ims[:,i+1])**2*configs.area_weight)
    #     img_mse.append(mse_i)
    #     print(i, mse_i)
    avg_mse = torch.mean((img_out[:,-output_length:]-test_ims[:,-output_length:])**2*configs.area_weight).cpu().numpy()
    print(f"{configs.save_file}, loss: {loss.mean()}, avg_mse: {avg_mse}")
    test_input_handle.next()
    

    # res_path = os.path.join(configs.gen_frm_dir, str(itr))
    # if not os.path.exists(res_path):
    #     os.makedirs(res_path)
    #     print('trainer.test function created path:', res_path)
    # else:
    #     print('trainer.test function found path:', res_path)
    # np.save(os.path.join(res_path,'true_data.npy'), test_ims.cpu().numpy())
    # np.save(os.path.join(res_path,'pred_data.npy'), img_out.cpu().numpy())
    if configs.upload_run:
        wandb.log({"Test mse": float(avg_mse)})
    return avg_mse


  