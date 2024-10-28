import os
import subprocess
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd
import torch

path = "/work/09012/haoli1/ls6/ERA5/"
train_data_names = "era5_train_1024002012_3_24hr.npz, era5_train_1001002015_3_24hr.npz, era5_train_1001002016_3_24hr.npz,era5_train_0827002021_3_24hr.npz"
train_data_files = train_data_names.split(',')
dataset = []
for i in range(len(train_data_files)):
    train_data_files[i] = path + train_data_files[i].strip()
    dataset.append(np.load(train_data_files[i]))

input_dataset = []
for i in range(len(dataset)):
    for key in dataset[i].keys():
        print(i, key, dataset[i][key].shape)
    input_dataset.append(dataset[i]['input_raw_data'])
input_data = np.concatenate(input_dataset, axis=0)
print(input_data.shape)
input_data =np.reshape(input_data, (*input_data.shape[:-2] , -1))
print(input_data.shape)

def gpu_rsvd(X, rank):
    Y = X @ torch.randn(size=(X.shape[1], rank), device='cuda')
    Q, _ = torch.linalg.qr(Y)
    B = Q.T @ X
    u_tilde, s, v = torch.linalg.svd(B, full_matrices = False)
    u = Q @ u_tilde
    torch.cuda.synchronize()
    return u, s, v

for i in range(3):
    a = torch.FloatTensor(input_data[:,i].T).to('cuda')
    U, s, Vh = gpu_rsvd(a, rank=a.shape[1])
    s = s.detach().cpu().numpy()
    print(f"Singular value {s[0]}, {s[-1]}")
    plt.figure()
    plt.plot(np.log(s))
    plt.savefig(path+f"{i}.png")