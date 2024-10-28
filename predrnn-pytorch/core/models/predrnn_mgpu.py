import torch
import numpy as np
import torch.nn as nn
from core.layers.SpatioTemporalLSTMCell_v2 import SpatioTemporalLSTMCell
import torch.nn.functional as F
from core.utils import preprocess
from core.utils.tsne import visualization
import sys
import pywt as pw
from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)\


class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()

        self.configs = configs
        self.visual = self.configs.visual
        self.visual_path = self.configs.visual_path
        self.skip_time = self.configs.skip_time
        self.wavelet = self.configs.wavelet
        self.loss_pos_neg = self.configs.output_length 

        self.num_layers = num_layers
        self.num_hidden = num_hidden
        # a magic number: the prescribed value for mean pressure.
        self.mean_press = 0.4480
        
        if not configs.time_step:
            self.mean_press_diff = nn.Parameter(torch.tensor([0, 1.39538188e-04,  1.99499756e-04,  1.47195304e-04, 3.02930121e-05, -1.19205533e-04, 
                                                -3.21148090e-04, -5.80101568e-04,-8.30103159e-04, -1.01917370e-03, -9.90518653e-04, -1.02008264e-03,
                                                -9.83945028e-04, -9.25706082e-04, -8.45840217e-04, -7.55326795e-04, -6.86238213e-04, -6.25248607e-04, 
                                                -5.60617360e-04, -4.71305105e-04, -3.53708443e-04, -2.18092631e-04, -3.34181112e-04, -1.63298334e-04]), requires_grad=False)
            self.freq = 24
            print(f"configs.time_step: 1 hour")
        elif configs.time_step == '6hrs':
            self.mean_press_diff = nn.Parameter(torch.tensor([0, -3.21148090e-04, -9.83945028e-04, -5.60617360e-04]), requires_grad=False)
            self.freq = 4
            print(f"configs.time_step: {configs.time_step}")
        # wavelet transformation function
        self.xfm = DWTForward(J=1, mode='zero', wave='db1')
        self.ifm = DWTInverse(mode='zero', wave='db1')

        cell_list = []

        self.patch_size = configs.patch_size
        height = configs.img_height // self.patch_size
        width = configs.img_width // self.patch_size
        # print(self.frame_channel)

        self.frame_channel = self.configs.img_channel*self.patch_size**2
        self.img_channel = self.configs.img_channel
        # print(f"self.configs.img_channel:{self.configs.img_channel}, self.frame_channel: {self.frame_channel}")
        self.img_chan_orig = self.img_channel
        self.loss_channel = self.img_channel*self.patch_size**2
        

        if self.configs.add_geopential:
            self.frame_channel += self.patch_size**2
            self.img_channel += 1
        if self.configs.add_land:
            self.frame_channel += self.patch_size**2
            self.img_channel += 1
        if self.configs.add_latitude:
            self.frame_channel += self.patch_size**2
            self.img_channel += 1

        self.cur_height = height
        self.cur_width = width
        self.img_height = configs.img_height
        self.img_width = configs.img_width
        
        if configs.use_weight ==1 :
            self.layer_weights = np.array([float(xi) for xi in configs.layer_weight.split(',')])
            if configs.is_WV ==0:
                if self.layer_weights.shape[0] != self.configs.img_channel:
                    print('error! number of channels and weigth should be the same')
                    print('weight length: '+str(self.layer_weights.shape[0]) +', number of channel: '+str(self.configs.img_channel))
                    sys.exit()
                self.layer_weights = np.repeat(self.layer_weights, self.patch_size**2)[np.newaxis,...]
            else:
                self.layer_weights = np.repeat(self.layer_weights, self.patch_size**2)[np.newaxis,...]
        else:
            self.layer_weights = np.ones((1))
        self.layer_weights = torch.FloatTensor(self.layer_weights).to(self.configs.device)

        self.area_weight = nn.Parameter(configs.area_weight, requires_grad=False)
        # self.area_weight = configs.area_weight
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i-1]
            cell_list.append(SpatioTemporalLSTMCell(in_channel, num_hidden[i], height, width, configs.filter_size,
                                                    configs.stride, configs.layer_norm))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.loss_channel, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        # shared adapter
        adapter_num_hidden = num_hidden[0]
        self.adapter = nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)
    

        
    def forward(self, frames_tensor, mask_true, extra_var=None, istrain=True):
        '''
        frames_tensor shape: [batch, length, channel, height, width]
        '''
        # print(f"Inside, frames_tensor shape:{frames_tensor.shape}, frames_tensor device: {frames_tensor.get_device()}")
        # print(f"Inside, self.area_weight device: {self.area_weight.get_device()}")

        tensor_device = frames_tensor.get_device()
        # self.mean_press = torch.mean(self.area_weight*frames_tensor[:,0,self.configs.layer_need_enhance])
        batch = frames_tensor.shape[0]
        mask_true = mask_true.contiguous()
        

        if self.configs.center_enhance:
            chan_en = frames_tensor[0,:,self.configs.layer_need_enhance,:,:]*(105000-98000)+98000
            self.zonal_mean = torch.mean(1/(chan_en[0,:,:]), dim=1) #get lattitude mean of the first time step
        
        if self.configs.is_WV:
            frames_tensor = self.img_to_wv(frames_tensor)
        else:
            if self.configs.center_enhance:
                frames_tensor = self.enhance(frames_tensor)
            frames_tensor = self.reshape_tensor(frames_tensor, self.patch_size)
        # print(f"frames_tensor shape: {frames_tensor.shape}, frames_tensor device:{frames_tensor.device}, mask_true shape: {mask_true.shape}")

        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []
        if self.visual:
            delta_c_visual = []
            delta_m_visual = []

        decouple_loss = []
        zeros = torch.zeros([batch, self.num_hidden[0], self.cur_height, self.cur_width]).to(tensor_device)
        for i in range(self.num_layers):
            # zeros = torch.zeros([batch, self.num_hidden[i], self.cur_height, self.cur_width]).to(tensor_device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)
        
        loss = 0
        memory = zeros
        next_frames = torch.zeros(batch, self.loss_pos_neg, self.loss_channel, self.cur_height, self.cur_width).to(tensor_device)
        net = torch.zeros(batch, self.frame_channel, self.cur_height, self.cur_width).to(tensor_device)
        if extra_var.any() is not None:
            extra_var = self.reshape_tensor(extra_var[:,None], self.patch_size)[:,0]
            # print(f"extra_var shape: {extra_var.shape}")
            net[:, self.loss_channel:] = extra_var
            # print(f"net shape: {net.shape}")
        # print(f"in the begining, next_frames shape:{next_frames.shape}")
        
        acc_gen_mean = 0
        for t in range(0, self.configs.total_length-1):
            if self.configs.reverse_scheduled_sampling == 1:
                # reverse schedule sampling
                if t < self.configs.total_length-self.loss_pos_neg:
                    net[:,:self.loss_channel] =  frames_tensor[:,t]
                else:
                    net[:,:self.loss_channel] = next_frames[:,t-self.loss_pos_neg]
            # else:
            #     # schedule sampling
            #     if t < self.configs.input_length:
            #         net = frames_tensor[:, t]
            #     else:
            #         net = mask_true[:, t-self.configs.input_length]*frames_tensor[:, t] + (1 - mask_true[:, t-self.configs.input_length])*next_frames[:,t-1]
            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory)
            delta_c_list[0] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list[0] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
            if self.visual:
                delta_c_visual.append(delta_c.view(delta_c.shape[0], delta_c.shape[1], -1))
                delta_m_visual.append(delta_m.view(delta_m.shape[0], delta_m.shape[1], -1))

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
                delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
                if self.visual:
                    delta_c_visual.append(delta_c.view(delta_c.shape[0], delta_c.shape[1], -1))
                    delta_m_visual.append(delta_m.view(delta_m.shape[0], delta_m.shape[1], -1))

            if t < self.configs.total_length-self.loss_pos_neg-1:
                x_gen = self.conv_last(h_t[self.num_layers - 1])
            else:
                next_frames[:,t+1-self.loss_pos_neg] = self.conv_last(h_t[self.num_layers - 1])
                x_gen = next_frames[:,t+1-self.loss_pos_neg:t+2-self.loss_pos_neg]
            
            if t >= self.configs.total_length-self.loss_pos_neg-1 and self.configs.press_constraint:
                if self.configs.is_WV:
                    # print(f"wv_to_img+de_enhance err:{torch.max(torch.abs(self.img_to_wv(self.enhance(self.de_enhance(self.wv_to_img(x_gen)))) - x_gen))}")
                    x_gen = self.wv_to_img(x_gen)
                    # print(f"img_to_wv err:{torch.max(torch.abs(self.wv_to_img(self.img_to_wv(x_gen)) - x_gen))}")
                    # print(f"after trans back to img, x_gen shape:{x_gen.shape}")
                    gen_mean = torch.mean(self.area_weight*x_gen[:,:,self.configs.layer_need_enhance])
                    acc_gen_mean += gen_mean
                    x_gen[:,:,self.configs.layer_need_enhance,:,:] = x_gen[:,:,self.configs.layer_need_enhance,:,:]+self.mean_press+self.mean_press_diff[(t+1)%self.freq]-gen_mean
                    # x_gen[:,:,self.configs.layer_need_enhance,:,:] = x_gen[:,:,self.configs.layer_need_enhance,:,:] + self.mean_press - gen_mean
                    # print(f"frames_tensor weighted mean press: {torch.mean(self.area_weight*x_gen[:,:,self.configs.layer_need_enhance])}")
                    x_gen= self.img_to_wv(x_gen)
                    # print(f"wv_to_img+de_enhance err:{torch.max(torch.abs(self.img_to_wv(self.wv_to_img(x_gen)) - x_gen))}")
                else:
                    x_gen = self.reshape_back_tensor(x_gen, self.patch_size)
                    if self.configs.center_enhance:
                        x_gen = self.de_enhance(x_gen)
                    # print(f"x_gen shape: {x_gen.shape}")
                    gen_mean = torch.mean(self.area_weight*x_gen[:,:,self.configs.layer_need_enhance])
                    acc_gen_mean += gen_mean
                    x_gen[:,:,self.configs.layer_need_enhance,:,:] = x_gen[:,:,self.configs.layer_need_enhance,:,:]+self.mean_press+self.mean_press_diff[(t+1)%self.freq]-gen_mean
                    # print(f"In the gen loop, gen_mean: {gen_mean}, mean_press_set: {self.mean_press}")
                    # x_gen[:,:,self.configs.layer_need_enhance,:,:] = x_gen[:,:,self.configs.layer_need_enhance,:,:]+self.mean_press-gen_mean
                    if self.configs.center_enhance:
                        x_gen = self.enhance(x_gen)
                    x_gen = self.reshape_tensor(x_gen, self.patch_size)

            if t >= self.configs.total_length-self.loss_pos_neg-1:
                next_frames[:,t+1-self.loss_pos_neg:t+2-self.loss_pos_neg] = x_gen
            
            # decoupling loss
            for i in range(0, self.num_layers):
                decouple_loss.append(torch.mean(torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))

        if self.visual:
            # visualization of delta_c and delta_m
            delta_c_visual = torch.stack(delta_c_visual, dim=0)
            delta_m_visual = torch.stack(delta_m_visual, dim=0)
            visualization(self.configs.total_length, self.num_layers, delta_c_visual, delta_m_visual, self.visual_path)
            self.visual = 0
        
        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        
        if self.configs.display_press_mean:
            if self.configs.is_WV:
                next_frames1 = self.wv_to_img(next_frames)
                gen_mean = torch.mean(self.area_weight*next_frames1[:,:,self.configs.layer_need_enhance])
            else:
                next_frames1 = self.reshape_back_tensor(next_frames, self.patch_size)
                # frames_tensor1 = self.reshape_back_tensor(frames_tensor[:,1:], self.patch_size)
                if self.configs.center_enhance:
                    next_frames1 = self.de_enhance(next_frames1)
                    # frames_tensor1 = self.de_enhance(frames_tensor1)
                gen_mean = torch.mean(self.area_weight*next_frames1[:,:,self.configs.layer_need_enhance])
                # frame_mean = torch.mean(self.area_weight*frames_tensor1[:,:,self.configs.layer_need_enhance])
            print(f"The modified gen p mean: {gen_mean}, gen p mean: {acc_gen_mean/(self.configs.total_length-1)}, mean p set: {self.mean_press}")

        if self.configs.weighted_loss and self.configs.is_WV != 1:
            if self.configs.is_WV == 2:
                next_frames = self.wv_to_img(next_frames)
                frames_tensor = self.wv_to_img(frames_tensor[:,-self.loss_pos_neg:])
                loss_pred = self.get_weighted_loss(next_frames[:,-self.loss_pos_neg:,:self.loss_channel], frames_tensor[:,-self.loss_pos_neg:,:self.loss_channel,:,:], reshape_back=False)
            else:
                loss_pred = self.get_weighted_loss(next_frames[:,-self.loss_pos_neg:,:self.loss_channel], frames_tensor[:,-self.loss_pos_neg:,:self.loss_channel,:,:])
        else:
            loss_pred = self.MSE_criterion(next_frames[:,-self.loss_pos_neg:,:self.loss_channel], frames_tensor[:,-self.loss_pos_neg:,:self.loss_channel,:,:])
        print(f"loss_pred:{loss_pred}, decouple_loss:{decouple_loss}")
        loss = loss_pred + self.configs.decouple_beta*decouple_loss
        if istrain:
            next_frames = None
            torch.cuda.empty_cache()
            return loss, loss_pred, decouple_loss
        else:
            if self.configs.is_WV == 1:
                next_frames = self.wv_to_img(next_frames)
            elif self.configs.is_WV == 0:
                next_frames = self.reshape_back_tensor(next_frames, self.patch_size)
                if self.configs.center_enhance:
                    next_frames = self.de_enhance(next_frames)
            torch.cuda.empty_cache()
            return next_frames, loss, loss_pred, decouple_loss
    

    def reshape_back_tensor(self, img_tensor, patch_size):
        assert 5 == img_tensor.ndim
        batch_size = img_tensor.shape[0]
        seq_length = img_tensor.shape[1]
        channels = img_tensor.shape[2]
        cur_height = img_tensor.shape[3]
        cur_width = img_tensor.shape[4]
        img_channels = channels // (patch_size*patch_size)
        img_tensor = torch.reshape(img_tensor, [batch_size, seq_length, img_channels, patch_size, patch_size,
                                        cur_height, cur_width])
        img_tensor = img_tensor.permute(0,1,2,5,3,6,4)
        return torch.reshape(img_tensor, [batch_size, seq_length, img_channels,
                                          cur_height*patch_size, cur_width*patch_size])

    def reshape_tensor(self, img_tensor, patch_size):
        assert 5 == img_tensor.ndim
        batch_size = img_tensor.shape[0]
        seq_length = img_tensor.shape[1]
        channels = img_tensor.shape[2]
        cur_height = img_tensor.shape[3]
        cur_width = img_tensor.shape[4]

        img_tensor = torch.reshape(img_tensor, [batch_size, seq_length, channels, 
                                                cur_height//patch_size, patch_size,
                                                cur_width//patch_size, patch_size])
        img_tensor = img_tensor.permute(0,1,2,4,6,3,5)
        return torch.reshape(img_tensor, [batch_size, seq_length, channels*patch_size*patch_size,
                                          cur_height//patch_size, cur_width//patch_size])


    def get_weighted_loss(self, pred_tensor, true_tensor, reshape_back=True):
        if reshape_back:
            pred_tensor = self.reshape_back_tensor(pred_tensor, self.patch_size)
            true_tensor = self.reshape_back_tensor(true_tensor, self.patch_size)
        # print(f"self.area_weight shape: {self.area_weight.shape}, pred_tensor shape: {pred_tensor.shape}, true_tensor shape:{true_tensor.shape}")
        return torch.mean((pred_tensor-true_tensor)**2*self.area_weight)

    def wv_to_img(self, frames_wv):
        '''
        frames_wv: [B, T, C, H, W]
        '''
        frames_wv = self.reshape_back_tensor(frames_wv, self.patch_size//2)
        B, T, C, H, W = frames_wv.shape
        frames_wv = torch.reshape(frames_wv, (B*T, C, H, W))

        Yl = frames_wv[:,0:self.img_channel,:,:]
        Yh1 = frames_wv[:,self.img_channel:2*self.img_channel,:,:]
        Yh2 = frames_wv[:,2*self.img_channel:3*self.img_channel,:,:]
        Yh3 = frames_wv[:,3*self.img_channel:4*self.img_channel,:,:]
        Yh = [torch.stack((Yh1, Yh2, Yh3), dim=2)]

        img_tensor = self.ifm((Yl, Yh))
        C, H, W = img_tensor.shape[1:]
        img_tensor = torch.reshape(img_tensor, (B, T, C, H, W))
        if self.configs.center_enhance:
            img_tensor = self.de_enhance(img_tensor)
        return img_tensor

    def img_to_wv(self, frames_tensor):
        '''
        img_tensor: [B, T, C, H, W]
        '''
        if self.configs.center_enhance:
            frames_tensor = self.enhance(frames_tensor)
        batch = frames_tensor.shape[0]
        seq_length = frames_tensor.shape[1]
        frames_tensor = torch.reshape(frames_tensor, (batch*seq_length, self.img_channel, self.configs.img_height, self.configs.img_width))
        Yl, Yh = self.xfm(frames_tensor)
        frames_tensor = torch.cat(((Yl, Yh[0][:,:,0], Yh[0][:,:,1], Yh[0][:,:,2])), dim=1)
        frames_tensor = torch.reshape(frames_tensor, (batch, seq_length, self.img_channel*4, self.img_height//2, self.img_width//2))
        frames_tensor = self.reshape_tensor(frames_tensor, self.patch_size//2)
        return frames_tensor
    
    def enhance(self, img_tensor):
        #center enhance
        chan_en = img_tensor[0,:,self.configs.layer_need_enhance,:,:]
        #unnormalize
        chan_en = chan_en *(105000 - 98000) + 98000
        chan_en = (1/chan_en) - self.zonal_mean[None,:,None]
        #re-normalize
        chan_en = (chan_en + 3e-7) / 7.7e-7
        img_tensor[0,:,self.configs.layer_need_enhance,:,:] = chan_en
        return img_tensor

    def de_enhance(self, img_tensor):
        chan_en = img_tensor[0,:,self.configs.layer_need_enhance,:,:]
        #unnormalize
        chan_en = chan_en * 7.7e-7 - 3e-7
        anomaly_zonal = 1/(chan_en + self.zonal_mean[None,:,None])
        #re-normalize
        chan_en = (anomaly_zonal - 98000) / (105000 - 98000)
        img_tensor[0,:,self.configs.layer_need_enhance,:,:] = chan_en
        return img_tensor