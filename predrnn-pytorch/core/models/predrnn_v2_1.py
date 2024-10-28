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
        self.skip_time = self.configs.skip_time
        self.wavelet = self.configs.wavelet
        if configs.is_WV:
            self.configs.img_channel = self.configs.img_channel * 10
        
        self.frame_channel = configs.patch_size * configs.patch_size * self.configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.last_patch = 5
        cell_list = []
        
        if configs.use_weight ==1 :
            self.layer_weights = np.array([float(xi) for xi in configs.layer_weight.split(',')])
            if configs.is_WV ==0:
                if self.layer_weights.shape[0] != self.configs.img_channel:
                    print('error! number of channels and weigth should be the same')
                    print('weight length: '+str(self.layer_weights.shape[0]) +', number of channel: '+str(self.configs.img_channel))
                    sys.exit()
                self.layer_weights = np.repeat(self.layer_weights, configs.patch_size * configs.patch_size)[np.newaxis,...]
            else:
                self.layer_weights = np.repeat(self.layer_weights, 3*(self.last_patch*4)**2 + \
                                                                   3*(self.last_patch*2)**2 + \
                                                                   4*(self.last_patch)**2)[np.newaxis,...]
        else:
            self.layer_weights = np.ones((1))
        #print(self.layer_weights.shape)
        
        self.layer_weights = torch.FloatTensor(self.layer_weights).to('cuda:1')
        height = configs.img_height // configs.patch_size
        width = configs.img_width // configs.patch_size
        #print(self.frame_channel)
        if configs.is_WV:
            height = int(configs.img_height/2) // (self.last_patch*4)
            width = int(configs.img_width/2) // (self.last_patch*4)
            self.frame_channel = int(self.configs.img_channel/10) * (3*(self.last_patch*4)**2 + \
                                       3*(self.last_patch*2)**2 + \
                                       4*(self.last_patch)**2)
        
        self.MSE_criterion = nn.MSELoss()
        #print(self.frame_channel)

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], height, width, configs.filter_size,
                                       configs.stride, configs.layer_norm).to("cuda:1")
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel, kernel_size=1, stride=1, padding=0,
                                   bias=False).to("cuda:2")
        # shared adapter
        adapter_num_hidden = num_hidden[0]
        self.adapter = nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, istrain=True):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous().to('cuda:1')
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous().to('cuda:1')

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]
        if self.configs.is_WV:
            curr_height = int(self.configs.img_height/2) // (self.last_patch*4)
            curr_width = int(self.configs.img_width/2) // (self.last_patch*4)
            frames_tensor = preprocess.reshape_patch_back(frames_tensor.detach().cpu().numpy(),self.configs.patch_size)
            #tcoeffs = pw.wavedec2(frames_tensor, 'db1', axes = (-3,-2), level=3)
            #(tcA3, (tcH3, tcV3, tcD3), (tcH2, tcV2, tcD2), (tcH1, tcV1, tcD1)) = tcoeffs
            tcoeffs = pw.wavedec2(frames_tensor, self.wavelet, axes = (-3,-2), level=1)
            tcA1, (tcH1, tcV1, tcD1) = tcoeffs
            #####Normalizing#####
            """
            norm_vect_H = [0.25]
            norm_vect_D = [0.15]
            norm_vect_A = [1.25]
            tcH3 = tcH3 / (4 * np.array(norm_vect_H)[np.newaxis,...])
            tcV3 = tcV3 / (4 * np.array(norm_vect_H)[np.newaxis,...])
            tcD3 = tcD3 / (4 * np.array(norm_vect_D)[np.newaxis,...])
            tcA3 = tcA3 / (4 * np.array(norm_vect_A)[np.newaxis,...])
            tcH2 = tcH2 / (2 * np.array(norm_vect_H)[np.newaxis,...])
            tcV2 = tcV2 / (2 * np.array(norm_vect_H)[np.newaxis,...])
            tcD2 = tcD2 / (2 * np.array(norm_vect_D)[np.newaxis,...])
            tcH1 = tcH1 / (np.array(norm_vect_H)[np.newaxis,...])
            tcV1 = tcV1 / (np.array(norm_vect_H)[np.newaxis,...])
            tcD1 = tcD1 / (np.array(norm_vect_D)[np.newaxis,...])
            #####Normalizing#####
            #####Reshaping#####
            tcA3_reshape = preprocess.reshape_patch(tcA3, self.last_patch)
            #tcA1_reshape = preprocess.reshape_patch(tcA1, self.last_patch*4)
            tcH3_reshape = preprocess.reshape_patch(tcH3, self.last_patch)
            tcV3_reshape = preprocess.reshape_patch(tcV3, self.last_patch)
            tcD3_reshape = preprocess.reshape_patch(tcD3, self.last_patch)
            tcH2_reshape = preprocess.reshape_patch(tcH2, self.last_patch*2)
            tcV2_reshape = preprocess.reshape_patch(tcV2, self.last_patch*2)
            tcD2_reshape = preprocess.reshape_patch(tcD2, self.last_patch*2)
            """
            tcH1_reshape = preprocess.reshape_patch(tcH1, self.last_patch*4)
            tcV1_reshape = preprocess.reshape_patch(tcV1, self.last_patch*4)
            tcD1_reshape = preprocess.reshape_patch(tcD1, self.last_patch*4)
            tcA1_reshape = preprocess.reshape_patch(tcA1, self.last_patch*4)
            #####Reshaping#####
            #frames = np.concatenate(((tcA3_reshape, tcH3_reshape, tcV3_reshape, tcD3_reshape,
            #                         tcH2_reshape, tcV2_reshape, tcD2_reshape,
            #                         tcH1_reshape, tcV1_reshape, tcD1_reshape)), axis=4)
            frames = np.concatenate(((tcA1_reshape, tcH1_reshape, tcV1_reshape, tcD1_reshape)), axis=4)
            frames = np.transpose(frames,(0, 1, 4, 2, 3))
            #frames = preprocess.reshape_patch(frames, self.configs.patch_size)
            frames = torch.FloatTensor(frames).to('cuda:1')
            if istrain:
                #delta_b = frames[:,1:,:,:,:] - frames[:,:-1,:,:,:]
                #frames_tensor = delta_b.detach().clone()
                #frames_tensor = frames_tensor.permute(0, 1, 3, 4, 2).contiguous()
                frames_tensor = frames.permute(0, 1, 3, 4, 2).contiguous()
            mask_true = mask_true[:,:,:,:curr_height,:curr_width]
            mask_true = torch.tile(mask_true[:,:,1:2,:,:],(1,1,self.frame_channel,1,1)).to("cuda:1")
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
                zeros = torch.zeros([batch, self.num_hidden[i], curr_height,curr_width]).to('cuda:1')
            else:
                zeros = torch.zeros([batch, self.num_hidden[i], height,width]).to('cuda:1')
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)
        
        loss = 0
        if self.configs.is_WV:
            memory = torch.zeros([batch, self.num_hidden[0], curr_height,curr_width]).to('cuda:1')
            next_frames = torch.empty(batch, self.configs.total_length - 1, 
                                          curr_height,curr_width, self.frame_channel).to('cuda:1')
        else:
            memory = torch.zeros([batch, self.num_hidden[0], height,width]).to('cuda:1')
            next_frames = torch.empty(batch, self.configs.total_length - 1, 
                                          height,width, self.frame_channel).to('cuda:1')
        for t in range(0,self.configs.total_length - 1):
            print(t)
            if self.configs.reverse_scheduled_sampling == 1:
                # reverse schedule sampling
                if t == 0:
                    net =  frames[:, t].to('cuda:1')
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen.to('cuda:1')
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
        #next_frames = next_frames[:,np.arange(0,self.configs.total_length),:,:,:]
        #frames_tensor = frames_tensor[:,np.arange(0,self.configs.total_length),:,:,:]
        if istrain:
            loss = self.MSE_criterion(next_frames.to('cuda:2')*self.layer_weights.to('cuda:2'), frames_tensor[:,1:,:,:,:].to('cuda:2')*self.layer_weights.to('cuda:2')) + \
                    self.configs.decouple_beta * decouple_loss.to('cuda:2')
            #loss = self.MSE_criterion(next_frames.to('cuda:2')*self.layer_weights.to('cuda:2'), frames_tensor[:,:,:,:,:].to('cuda:2')*self.layer_weights.to('cuda:2')) + \
            #        self.configs.decouple_beta * decouple_loss.to('cuda:2')
            next_frames = None
        else:
            if self.configs.is_WV:
                #next_frames = preprocess.reshape_patch_back(next_frames.detach().cpu().numpy(), 
                #                                              self.configs.patch_size)
                #next_frames = next_frames.permute(0, 1, 4, 2, 3).contiguous()
                next_frames = next_frames.detach().cpu().numpy()
                #print(next_frames.shape)
                curr_img_channel = int(self.configs.img_channel/10)
                curr_position = 0
                next_position = curr_img_channel*(self.last_patch*4)**2
                """
                tcA3_next = next_frames[...,int(curr_position):int(next_position)]
                tcA3_next = preprocess.reshape_patch_back(tcA3_next, self.last_patch)
                tcA3_next = tcA3_next * (4 * np.array(norm_vect_A)[np.newaxis,...])
                curr_position = next_position
                next_position += curr_img_channel*(self.last_patch)**2
                
                tcH3_next = next_frames[...,int(curr_position):int(next_position)]
                tcH3_next = preprocess.reshape_patch_back(tcH3_next, self.last_patch)
                tcH3_next = tcH3_next * (4 * np.array(norm_vect_H)[np.newaxis,...])
                curr_position = next_position
                next_position += curr_img_channel*(self.last_patch)**2
                
                tcV3_next = next_frames[...,int(curr_position):int(next_position)]
                tcV3_next = preprocess.reshape_patch_back(tcV3_next, self.last_patch)
                tcV3_next = tcV3_next * (4 * np.array(norm_vect_H)[np.newaxis,...])
                curr_position = next_position
                next_position += curr_img_channel*(self.last_patch)**2
                
                tcD3_next = next_frames[...,int(curr_position):int(next_position)]
                tcD3_next = preprocess.reshape_patch_back(tcD3_next, self.last_patch)
                tcD3_next = tcD3_next * (4 * np.array(norm_vect_D)[np.newaxis,...])
                curr_position = next_position
                next_position += curr_img_channel*(self.last_patch*2)**2
                
                tcH2_next = next_frames[...,int(curr_position):int(next_position)]
                tcH2_next = preprocess.reshape_patch_back(tcH2_next, self.last_patch*2)
                tcH2_next = tcH2_next * (2 * np.array(norm_vect_H)[np.newaxis,...])
                curr_position = next_position
                next_position += curr_img_channel*(self.last_patch*2)**2
                
                tcV2_next = next_frames[...,int(curr_position):int(next_position)]
                tcV2_next = preprocess.reshape_patch_back(tcV2_next, self.last_patch*2)
                tcV2_next = tcV2_next * (2 * np.array(norm_vect_H)[np.newaxis,...])
                curr_position = next_position
                next_position += curr_img_channel*(self.last_patch*2)**2
                
                tcD2_next = next_frames[...,int(curr_position):int(next_position)]
                tcD2_next = preprocess.reshape_patch_back(tcD2_next, self.last_patch*2)
                tcD2_next = tcD2_next * (2 * np.array(norm_vect_D)[np.newaxis,...])
                curr_position = next_position
                next_position += curr_img_channel*(self.last_patch*4)**2
                """
                tcA1_next = next_frames[...,int(curr_position):int(next_position)]
                tcA1_next = preprocess.reshape_patch_back(tcA1_next, self.last_patch*4)
                #tcH1_next = tcH1_next * (np.array(norm_vect_H)[np.newaxis,...])
                curr_position = next_position
                next_position += curr_img_channel*(self.last_patch*4)**2
                
                tcH1_next = next_frames[...,int(curr_position):int(next_position)]
                tcH1_next = preprocess.reshape_patch_back(tcH1_next, self.last_patch*4)
                #tcH1_next = tcH1_next * (np.array(norm_vect_H)[np.newaxis,...])
                curr_position = next_position
                next_position += curr_img_channel*(self.last_patch*4)**2
                
                tcV1_next = next_frames[...,int(curr_position):int(next_position)]
                tcV1_next = preprocess.reshape_patch_back(tcV1_next, self.last_patch*4)
                #tcV1_next = tcV1_next * (np.array(norm_vect_H)[np.newaxis,...])
                curr_position = next_position
                next_position += curr_img_channel*(self.last_patch*4)**2
                
                tcD1_next = next_frames[...,int(curr_position):int(next_position)]
                tcD1_next = preprocess.reshape_patch_back(tcD1_next, self.last_patch*4)
                #tcD1_next = tcD1_next * (np.array(norm_vect_D)[np.newaxis,...])
                #print(tcA1_next.shape,tcH1_next.shape,tcV1_next.shape,tcD1_next.shape) 
                srcoeffs = (tcA1_next,
                        (tcH1_next, tcV1_next, tcD1_next))
                ###Wavelet transform
                """
                srcoeffs = (0.5*next_frames[...,:int(self.frame_channel/4)] + \
                            0.5*prev_frames[...,:int(self.frame_channel/4)],
                            (0.5*next_frames[...,int(self.frame_channel/4):int(self.frame_channel/4)*2] + \
                             0.5*prev_frames[...,int(self.frame_channel/4):int(self.frame_channel/4)*2],
                             0.5*next_frames[...,int(self.frame_channel/4)*2:int(self.frame_channel/4)*3] + \
                             0.5*prev_frames[...,int(self.frame_channel/4)*2:int(self.frame_channel/4)*3],
                             0.5*next_frames[...,int(self.frame_channel/4)*3:int(self.frame_channel)] + \
                            0.5*prev_frames[...,int(self.frame_channel/4)*3:int(self.frame_channel)]))
                
                srcoeffs = (next_frames[...,:int(self.frame_channel/4)],
                            (next_frames[...,int(self.frame_channel/4):int(self.frame_channel/4)*2],
                             next_frames[...,int(self.frame_channel/4)*2:int(self.frame_channel/4)*3],
                             next_frames[...,int(self.frame_channel/4)*3:int(self.frame_channel)]))
                #srcoeffs = (tcA[:, 1:], (tcH[:, 1:], tcV[:, 1:], tcD[:, 1:]))
                print(tcA.shape)
                """
                next_frames = pw.waverec2(srcoeffs, self.wavelet, axes = (-3,-2))
                next_frames = preprocess.reshape_patch(next_frames, self.configs.patch_size)
                next_frames = torch.FloatTensor(next_frames).to(self.configs.device)
        return next_frames, loss