import os.path
import datetime
import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from core.utils import preprocess, metrics
import lpips
import torch

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
    
    if cost < configs.curr_best_loss:
        print('current best loss: '+str(np.round(cost,6)))
        configs.curr_best_loss = cost
        model.save('best')
    
    if itr % configs.display_interval == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
        print('training loss: ' + str(cost))


def test(model, test_input_handle, configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
    test_input_handle.begin(do_shuffle=False)
    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.mkdir(res_path)
    avg_mse = 0
    batch_id = 0
    img_mse, ssim, psnr = [], [], []
    lp = []
    for i in range(configs.total_length - configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        lp.append(0)

    # reverse schedule sampling
    if configs.reverse_scheduled_sampling == 1:
        mask_input = 1
    else:
        mask_input = configs.input_length
    
    real_input_flag = np.zeros(
            (configs.batch_size,
             configs.total_length - mask_input - 1,
             configs.img_height // configs.patch_size,
             configs.img_width // configs.patch_size,
             configs.patch_size ** 2 * configs.img_channel))

    if configs.reverse_scheduled_sampling == 1:
        real_input_flag[:, :configs.input_length - 1, :, :] = 1.0

    while (test_input_handle.no_batch_left() == False):
        batch_id = batch_id + 1
        test_ims = test_input_handle.get_batch()
        test_ims = test_ims[:, :, :, :, :configs.img_channel]
        #test_ims = test_ims * layer_weights
        
        #center enhance
        if configs.center_enhance:
            enh_ims = test_ims.copy()
            layer_ims = enh_ims[0,:,:,:,configs.layer_need_enhance]
            #unnormalize
            layer_ims = layer_ims *(105000 - 98000) + 98000
            zonal_mean = np.mean(1/(layer_ims[0,:,:]), axis=1) #get lattitude mean of the first time step
            anomaly_zonal = (1/layer_ims) - zonal_mean[None,:,None]
            #re-normalize
            layer_ims = (anomaly_zonal + 3e-7) / 7.7e-7
            enh_ims[0,:,:,:,configs.layer_need_enhance] = layer_ims
            test_ims = enh_ims.copy()
        
        #append output length to test_ims
        output_length = configs.total_length - configs.input_length
        curr_shapes = test_ims.shape
        zero_pad_arr = np.zeros((curr_shapes[0],output_length,
                                 curr_shapes[2],curr_shapes[3],curr_shapes[4]))
        test_ims_pad = np.concatenate([test_ims, zero_pad_arr], axis=1)
        ##############
        test_dat = preprocess.reshape_patch(test_ims_pad, configs.patch_size)
        img_gen = model.test(test_dat, real_input_flag)
        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        output_length = configs.total_length - configs.input_length 
        img_out = img_gen.copy()
        ##############
        #center de-enhance
        if configs.center_enhance:
            enh_ims = img_out.copy()
            layer_ims = enh_ims[0,:,:,:,configs.layer_need_enhance]
            #unnormalize
            layer_ims = layer_ims * 7.7e-7 - 3e-7
            anomaly_zonal = 1/(layer_ims + zonal_mean[None,:,None])
            #re-normalize
            layer_ims = (anomaly_zonal - 98000) / (105000 - 98000)
            enh_ims[0,:,:,:,configs.layer_need_enhance] = layer_ims
            img_out = enh_ims.copy()
            #de-enhance for input images as well
            enh_ims = test_ims.copy()
            layer_ims = enh_ims[0,:,:,:,configs.layer_need_enhance]
            #unnormalize
            layer_ims = layer_ims * 7.7e-7 - 3e-7
            anomaly_zonal = 1/(layer_ims + zonal_mean[None,:,None])
            #re-normalize
            layer_ims = (anomaly_zonal - 98000) / (105000 - 98000)
            enh_ims[0,:,:,:,configs.layer_need_enhance] = layer_ims
            test_ims = enh_ims.copy()
        
        #img_out = img_out/layer_weights
        #test_ims = test_ims/layer_weights

        # MSE per frame
        for i in range(output_length):
            x = test_ims[:, i + configs.input_length, :, :, :]
            gx = img_out[:, i, :, :, :]
            gx = np.maximum(gx, 0)
            gx = np.minimum(gx, 1)
            mse = np.square(x - gx).sum()
            img_mse[i] += mse
            avg_mse += mse
            # cal lpips
            img_x = np.zeros([configs.batch_size, 3, configs.img_height, configs.img_width])
            if configs.img_channel == 3:
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 1]
                img_x[:, 2, :, :] = x[:, :, :, 2]
            else:
                img_x[:, 0, :, :] = x[:, :, :, 0]
                img_x[:, 1, :, :] = x[:, :, :, 0]
                img_x[:, 2, :, :] = x[:, :, :, 0]
            img_x = torch.FloatTensor(img_x)
            img_gx = np.zeros([configs.batch_size, 3, configs.img_height, configs.img_width])
            if configs.img_channel == 3:
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 1]
                img_gx[:, 2, :, :] = gx[:, :, :, 2]
            else:
                img_gx[:, 0, :, :] = gx[:, :, :, 0]
                img_gx[:, 1, :, :] = gx[:, :, :, 0]
                img_gx[:, 2, :, :] = gx[:, :, :, 0]
            img_gx = torch.FloatTensor(img_gx)
            lp_loss = loss_fn_alex(img_x, img_gx)
            lp[i] += torch.mean(lp_loss).item()

            real_frm = np.uint8(x * 255)
            pred_frm = np.uint8(gx * 255)

            psnr[i] += metrics.batch_psnr(pred_frm, real_frm)
            for b in range(configs.batch_size):
                score, _ = compare_ssim(pred_frm[b], real_frm[b], full=True, multichannel=True)
                ssim[i] += score

        # save prediction examples
        if batch_id <= configs.num_save_samples:
            path = os.path.join(res_path, str(batch_id))
            os.mkdir(path)
            np.save(os.path.join(path,'gt.npy'), test_ims)
            np.save(os.path.join(path,'pd.npy'), img_out)
            """
            for i in range(configs.total_length):
                name = 'gt' + str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8(test_ims[0, i, :, :, :] * 255)
                cv2.imwrite(file_name, img_gt)
            for i in range(output_length):
                name = 'pd' + str(i + 1 + configs.input_length) + '.png'
                file_name = os.path.join(path, name)
                img_pd = img_out[0, i, :, :, :]
                img_pd = np.maximum(img_pd, 0)
                img_pd = np.minimum(img_pd, 1)
                img_pd = np.uint8(img_pd * 255)
                cv2.imwrite(file_name, img_pd)
            """
        test_input_handle.next()

    avg_mse = avg_mse / (batch_id * configs.batch_size)
    print('mse per seq: ' + str(avg_mse))
    for i in range(configs.total_length - configs.input_length):
        print(img_mse[i] / (batch_id * configs.batch_size))

    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
    print('ssim per frame: ' + str(np.mean(ssim)))
    for i in range(configs.total_length - configs.input_length):
        print(ssim[i])

    psnr = np.asarray(psnr, dtype=np.float32) / batch_id
    print('psnr per frame: ' + str(np.mean(psnr)))
    for i in range(configs.total_length - configs.input_length):
        print(psnr[i])

    lp = np.asarray(lp, dtype=np.float32) / batch_id
    print('lpips per frame: ' + str(np.mean(lp)))
    for i in range(configs.total_length - configs.input_length):
        print(lp[i])
