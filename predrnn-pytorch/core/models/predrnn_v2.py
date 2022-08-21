__author__ = 'yunbo'

import torch
import numpy as np
import torch.nn as nn
from core.layers.SpatioTemporalLSTMCell_v2 import SpatioTemporalLSTMCell
import torch.nn.functional as F
from core.utils import preprocess
from core.utils.tsne import visualization
import sys
import pywt as pw

class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()

        self.configs = configs
        self.visual = self.configs.visual
        self.visual_path = self.configs.visual_path
        if configs.is_WV:
            self.configs.img_channel = self.configs.img_channel * 4

        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []
        
        if configs.use_weight ==1 :
            self.layer_weights = np.array([float(xi) for xi in configs.layer_weight.split(',')])
            if configs.is_WV ==0:
                if self.layer_weights.shape[0] != configs.img_channel:
                    print('error! number of channels and weigth should be the same')
                    print('weight length: '+str(self.layer_weights.shape[0]) +', number of channel: '+str(configs.img_channel))
                    sys.exit()
                self.layer_weights = np.repeat(self.layer_weights, configs.patch_size * configs.patch_size)[np.newaxis,...]
            else:
                self.layer_weights = np.repeat(self.layer_weights, configs.patch_size * configs.patch_size *4)[np.newaxis,...]
        else:
            self.layer_weights = np.ones((1))
        #print(self.layer_weights.shape)
        
        self.layer_weights = torch.FloatTensor(self.layer_weights).to(self.configs.device)
        height = configs.img_height // configs.patch_size
        width = configs.img_width // configs.patch_size
        
        if configs.is_WV:
            height, width = int(height/2), int(width/2)
        
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], height, width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel, kernel_size=1, stride=1, padding=0,
                                   bias=False)
        # shared adapter
        adapter_num_hidden = num_hidden[0]
        self.adapter = nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, istrain=True):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]
        if self.configs.is_WV:
            tcoeffs = pw.dwt2(frames.detach().cpu().numpy(), 'db1', axes = (-2,-1))
            tcA, (tcH, tcV, tcD) = tcoeffs
            frames = np.concatenate(((tcA, tcH, tcV, tcD)), axis=2)
            #frames = preprocess.reshape_patch(frames, self.configs.patch_size)
            frames = torch.FloatTensor(frames).to(self.configs.device)
            if istrain:
                delta_b = frames[:,1:,:,:,:] - frames[:,:-1,:,:,:]
                frames_tensor = delta_b.detach().clone()
                frames_tensor = frames.permute(0, 1, 3, 4, 2).contiguous()
            mask_true = mask_true[:,:,:,:int(height/2),:int(width/2)]
            """
            tcA = tcA[:,1:,:,:,:]
            tcH = tcH[:,1:,:,:,:]
            tcV = tcV[:,1:,:,:,:]
            tcD = tcD[:,1:,:,:,:]
            """
        #print(frames.shape)
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []
        if self.visual:
            delta_c_visual = []
            delta_m_visual = []

        decouple_loss = []

        for i in range(self.num_layers):
            if self.configs.is_WV:
                zeros = torch.zeros([batch, self.num_hidden[i], int(height/2),int(width/2)]).to(self.configs.device)
            else:
                zeros = torch.zeros([batch, self.num_hidden[i], height,width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)
        
        loss = 0
        if self.configs.is_WV:
            memory = torch.zeros([batch, self.num_hidden[0], int(height/2),int(width/2)]).to(self.configs.device)
            next_frames = torch.empty(batch, self.configs.total_length - 1, 
                                          int(height/2),int(width/2), self.frame_channel).to(self.configs.device)
        else:
            memory = torch.zeros([batch, self.num_hidden[0], height,width]).to(self.configs.device)
            next_frames = torch.empty(batch, self.configs.total_length - 1, 
                                          height,width, self.frame_channel).to(self.configs.device)
        for t in range(self.configs.total_length - 1):
            # print(t)
            if self.configs.reverse_scheduled_sampling == 1:
                # reverse schedule sampling
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                # schedule sampling
                if t < self.configs.input_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                          (1 - mask_true[:, t - self.configs.input_length]) * x_gen

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

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames[:,t,:,:,:] = x_gen.permute(0, 2, 3, 1).to(self.configs.device)
            # decoupling loss
            for i in range(0, self.num_layers):
                decouple_loss.append(
                    torch.mean(torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))

        if self.visual:
            # visualization of delta_c and delta_m
            delta_c_visual = torch.stack(delta_c_visual, dim=0)
            delta_m_visual = torch.stack(delta_m_visual, dim=0)
            visualization(self.configs.total_length, self.num_layers, delta_c_visual, delta_m_visual, self.visual_path)
            self.visual = 0
        # next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        if istrain:
            loss = self.MSE_criterion(next_frames*self.layer_weights, frames_tensor[:,1:]*self.layer_weights) + \
                    self.configs.decouple_beta * decouple_loss
            next_frames = None
            torch.cuda.empty_cache()
        else:
            if self.configs.is_WV:
                #next_frames = preprocess.reshape_patch_back(next_frames.detach().cpu().numpy(), 
                #                                              self.configs.patch_size)
                prev_frames = frames[:,1:].permute(0, 1, 3, 4, 2).contiguous()
                prev_frames = prev_frames.detach().cpu().numpy()
                next_frames = next_frames.detach().cpu().numpy()
                ###Wavelet transform
                srcoeffs = (0.5*next_frames[...,:int(self.frame_channel/4)] + \
                            0.5*prev_frames[...,:int(self.frame_channel/4)],
                            (0.5*next_frames[...,int(self.frame_channel/4):int(self.frame_channel/4)*2] + \
                             0.5*prev_frames[...,int(self.frame_channel/4):int(self.frame_channel/4)*2],
                             0.5*next_frames[...,int(self.frame_channel/4)*2:int(self.frame_channel/4)*3] + \
                             0.5*prev_frames[...,int(self.frame_channel/4)*2:int(self.frame_channel/4)*3],
                             0.5*next_frames[...,int(self.frame_channel/4)*3:int(self.frame_channel)] + \
                            0.5*prev_frames[...,int(self.frame_channel/4)*3:int(self.frame_channel)]))
                """
                srcoeffs = (next_frames[...,:int(self.frame_channel/4)],
                            (next_frames[...,int(self.frame_channel/4):int(self.frame_channel/4)*2],
                             next_frames[...,int(self.frame_channel/4)*2:int(self.frame_channel/4)*3],
                             next_frames[...,int(self.frame_channel/4)*3:int(self.frame_channel)]))
                #srcoeffs = (tcA[:, 1:], (tcH[:, 1:], tcV[:, 1:], tcD[:, 1:]))
                print(tcA.shape)
                """
                next_frames = pw.idwt2(srcoeffs, 'db1', axes = (-3,-2))
                #next_frames = preprocess.reshape_patch(next_frames, self.configs.patch_size)
                next_frames = torch.FloatTensor(next_frames).to(self.configs.device)
        return next_frames, loss
