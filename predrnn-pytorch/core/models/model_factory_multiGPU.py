import os
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from core.models import predrnn, predrnn_v2, predrnn_mgpu, action_cond_predrnn, action_cond_predrnn_v2
import wandb
import gc


class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            'predrnn': predrnn.RNN,
            'predrnn_v2': predrnn_mgpu.RNN,
            'action_cond_predrnn': action_cond_predrnn.RNN,
            'action_cond_predrnn_v2': action_cond_predrnn_v2.RNN,
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)
        devices = [i for i in range(configs.gpu_num)]
        self.network = nn.DataParallel(self.network, device_ids=devices).to(self.configs.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=configs.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9998)
        if self.configs.upload_run:
            self.upload_wandb()
    
    
    def upload_wandb(self):
        # Uploading to wandb
        run_name = '_'.join((self.configs.save_file, self.configs.run_name))
        wandb.init(project=self.configs.project, name=run_name)
        wandb.config.model_name = self.configs.model_name
        wandb.config.opt = self.configs.opt
        wandb.config.lr = self.configs.lr
        wandb.config.batch_size = 1


    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model'+'_'+str(itr)+'.ckpt')
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path, map_location=torch.device(self.configs.device))
        self.network.load_state_dict(stats['net_param'])

    def train(self, frames, mask, extra_var, istrain=True):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        if extra_var.all() is not None:
            extra_var = torch.FloatTensor(extra_var).to(self.configs.device)        

        self.optimizer.zero_grad()
        # print(f"Outside, frames_tensor shape:{frames_tensor.shape}, mask_tensor shape: {mask_tensor.shape}")
        loss, _, _ = self.network(frames_tensor, mask_tensor, extra_var, istrain=True)
        #loss = loss.to('cpu')
        del frames_tensor
        del mask_tensor
        del extra_var
        gc.collect() 
        torch.cuda.empty_cache()
        loss.mean().backward()
        self.optimizer.step()
        self.scheduler.step()
        #del loss
        gc.collect() 
        torch.cuda.empty_cache()
        return loss.detach().cpu().numpy()

    def test(self, frames_tensor, mask_tensor, extra_var_tensor):
        input_length = self.configs.input_length
        total_length = self.configs.total_length
        output_length = total_length - input_length
        B, T, C, W, H = frames_tensor.shape
        if self.configs.concurent_step > 1:
            # I have not modified it to make it work.
            # pred_tensor = torch.zeros((B, T+output_length*self.configs.concurent_step, C, W, H)).to(self.configs.device)
            #pred_tensor = frames_tensor.clone()
            print(f"frames_tensor shape:{frames_tensor.shape}")
            for i in range(self.configs.concurent_step):
                with torch.no_grad():
                    next_frames, loss,  _, _ = self.network(frames_tensor=frames_tensor[:, output_length*i:output_length*i+total_length,:,:,:], 
                                                            mask_true=mask_tensor,                                                                     extra_var=extra_var_tensor,
                                                            istrain=False)
                print(f"next_frames shape: {next_frames.shape}")
                frames_tensor[:next_frames.shape[0], output_length*i+input_length:output_length*i+total_length,:,:,:] = next_frames[:,-output_length:,:,:,:]
                del next_frames
                torch.cuda.empty_cache()
            return frames_tensor, loss
        else:
            with torch.no_grad():
                # print(f"frames_tensor: {frames_tensor.shape}, mask_tensor: {mask_tensor.shape}")
                torch.cuda.empty_cache()
                next_frames, loss, _, _ = self.network(frames_tensor, mask_tensor, extra_var_tensor, istrain=False)
                avg_mse = torch.mean((next_frames[:,-output_length:]-frames_tensor[:,-output_length:])**2*self.configs.area_weight).detach().cpu().numpy()
                torch.cuda.empty_cache()
            frames_tensor[:next_frames.shape[0], -output_length:,:,:,:] = next_frames[:,-output_length:,:,:,:]
            print(f"next_frames shape: {next_frames.shape}")
            return frames_tensor, loss
        
