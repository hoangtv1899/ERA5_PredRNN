import os
from time import sleep
from osgeo import gdal
import numpy as np
from glob import glob
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta
import pygrib
import numpy
import matplotlib
import torch
from pytorch_wavelets import DWTForward, DWTInverse # (or import DWT, IDWT)\
import pywt as pw

def wv_to_img(frames_wv, cut, wave='db1'):
    xfm = DWTForward(J=1, mode='zero', wave=wave)
    ifm = DWTInverse(mode='zero', wave=wave)
    Yl = frames_wv[:,0:cut,:,:]
    tcA1_next = preprocess.reshape_patch_back(tcA1_next, self.last_patch*4)

                next_frames = np.transpose(next_frames, (0,1,3,4,2))
                #print(next_frames.shape)
                self.img_channel = int(self.configs.img_channel/10)
                curr_position = 0
                next_position = self.img_channel*(self.last_patch*4)**2
                
                tcA1_next = next_frames[...,int(curr_position):int(next_position)]
                tcA1_next = preprocess.reshape_patch_back(tcA1_next, self.last_patch*4)
                #tcH1_next = tcH1_next * (np.array(norm_vect_H)[np.newaxis,...])
                curr_position = next_position
                next_position += self.img_channel*(self.last_patch*4)**2
                
                tcH1_next = next_frames[...,int(curr_position):int(next_position)]
                tcH1_next = preprocess.reshape_patch_back(tcH1_next, self.last_patch*4)
                #tcH1_next = tcH1_next * (np.array(norm_vect_H)[np.newaxis,...])
                curr_position = next_position
                next_position += self.img_channel*(self.last_patch*4)**2
                
                tcV1_next = next_frames[...,int(curr_position):int(next_position)]
                tcV1_next = preprocess.reshape_patch_back(tcV1_next, self.last_patch*4)
                #tcV1_next = tcV1_next * (np.array(norm_vect_H)[np.newaxis,...])
                curr_position = next_position
                next_position += self.img_channel*(self.last_patch*4)**2
                
                tcD1_next = next_frames[...,int(curr_position):int(next_position)]
                tcD1_next = preprocess.reshape_patch_back(tcD1_next, self.last_patch*4)
                #tcD1_next = tcD1_next * (np.array(norm_vect_D)[np.newaxis,...])
                #print(tcA1_next.shape,tcH1_next.shape,tcV1_next.shape,tcD1_next.shape) 
                srcoeffs = (tcA1_next, (tcH1_next, tcV1_next, tcD1_next))

                next_frames = pw.waverec2(srcoeffs, self.wavelet, axes = (-3,-2))
                next_frames = preprocess.reshape_patch(next_frames, self.configs.patch_size)
                next_frames = torch.FloatTensor(next_frames).to(self.configs.device)