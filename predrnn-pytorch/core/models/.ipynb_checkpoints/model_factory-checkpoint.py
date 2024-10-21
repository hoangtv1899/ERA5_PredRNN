import os
import numpy as np
import torch
from torch.optim import Adam
from core.models import predrnn, predrnn_v2, action_cond_predrnn, action_cond_predrnn_v2

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
            self.network = Network(self.num_layers, self.num_hidden, configs).to('cuda:1')
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt'+'-'+str(itr))
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path, map_location=torch.device('cuda:1'))
        self.network.load_state_dict(stats['net_param'])

    def train(self, frames, mask, istrain=True):
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        self.optimizer.zero_grad()
        next_frames, loss = self.network(frames_tensor, mask_tensor,istrain=istrain)
        loss.backward()
        del next_frames
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def test(self, frames, mask, istrain=False):
        input_length = self.configs.input_length
        total_length = self.configs.total_length
        output_length = total_length - input_length
        frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        #frames_tensor[:,total_length*2:,:,:,:] = 0
        mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        final_next_frames = []
        for i in range(self.configs.concurent_step):
            #print(i)
            with torch.no_grad():
                next_frames, _ = self.network(frames_tensor[:,input_length*i:input_length*i+total_length,:,:,:], 
                                          mask_tensor,
                                          istrain=istrain)
            frames_tensor[:,input_length*i+total_length - output_length:\
                          input_length*i+total_length,:,:,:] = next_frames[:,-output_length:,:,:,:]
            final_next_frames.append(next_frames[:,-output_length:,:,:,:].detach().cpu().numpy())
            del next_frames
            torch.cuda.empty_cache()
        return np.hstack(final_next_frames)
