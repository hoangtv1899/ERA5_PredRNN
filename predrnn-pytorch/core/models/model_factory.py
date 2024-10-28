import os
import numpy as np
import torch
from torch.optim import Adam
from core.models import predrnn, predrnn_v2, action_cond_predrnn, action_cond_predrnn_v2
import wandb
import gc

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            'predrnn': predrnn.RNN,
            'predrnn_v2': predrnn_v2.RNN,
            'action_cond_predrnn': action_cond_predrnn.RNN,
            'action_cond_predrnn_v2': action_cond_predrnn_v2.RNN,
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(self.configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=configs.lr)
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

    def train(self, frames, mask, istrain=True):
        gc.collect()
        torch.cuda.empty_cache()
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        self.optimizer.zero_grad()
        loss, loss_pred, decouple_loss = self.network(frames_tensor, mask_tensor,istrain=istrain)
        torch.cuda.empty_cache()
        loss.backward()
        self.optimizer.step()
        if self.configs.upload_run:
            wandb.log({"Total Loss": float(loss), "Pred Loss": loss_pred, 'Decop Loss': decouple_loss})
        return loss.detach().cpu().numpy()

    def test(self, frames_tensor, mask_tensor):
        input_length = self.configs.input_length
        total_length = self.configs.total_length
        output_length = total_length - input_length
        final_next_frames = []
        if self.configs.concurent_step > 1:
            # I have not modified it to make it work.
            for i in range(self.configs.concurent_step):
                print(i)
                with torch.no_grad():
                    next_frames, loss, loss_pred, decouple_loss= self.network(frames_tensor[:,input_length*i:input_length*i+total_length,:,:,:], mask_tensor, istrain=False)
                print(f"next_frames shape:{next_frames.shape}, frames_tensor shape:{frames_tensor.shape}")
                frames_tensor[:,input_length*i+input_length:input_length*i+total_length,:,:,:] = next_frames[:,-output_length:,:,:,:]
                final_next_frames.append(next_frames[:,-output_length:,:,:,:].detach().cpu().numpy())
                del next_frames
                torch.cuda.empty_cache()
        else:
            with torch.no_grad():
                next_frames, loss, loss_pred, decouple_loss = self.network(frames_tensor, mask_tensor, istrain=False)

        return next_frames, loss, loss_pred, decouple_loss
