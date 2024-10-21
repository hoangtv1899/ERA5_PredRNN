import os
import sys
import xarray as xr
import numpy as np
#import xesmf as xe
import pandas as pd
from glob import glob
from datetime import datetime, timedelta

import torch
import onnx
import onnxruntime as ort

# The directory of your input and output data
input_data_dir = '/scratch/08589/hvtran/Pangu-Weather/input_2023'
#model_24 = onnx.load('pangu_weather_24.onnx')
model_1 = onnx.load('pangu_weather_1.onnx')
model_6 = onnx.load('pangu_weather_6.onnx')

options = ort.SessionOptions()
options.enable_cpu_mem_arena=False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
# Increase the number for faster inference and more memory consumption
options.intra_op_num_threads = 3

# Set the behavier of cuda provider
cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

ort_session_1 = ort.InferenceSession('pangu_weather_1.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])
ort_session_6 = ort.InferenceSession('pangu_weather_6.onnx', sess_options=options, providers=[('CUDAExecutionProvider', cuda_provider_options)])


# Load the upper-air numpy arrays
input_surface_files = sorted(glob(input_data_dir+'/*surface*.npy'))
input_upper_files = sorted(glob(input_data_dir+'/*upper*.npy'))

for ii,filei in enumerate(input_surface_files):
    print(filei)
    input = np.load(input_upper_files[ii]).astype(np.float32)
    input_surface = np.load(filei).astype(np.float32)
    
    input_1, input_surface_1 = input.copy(), input_surface.copy()
    print('1 hour')
    for i in range(24):
        output_1, output_surface_1 = ort_session_1.run(None, {'input':input_1, 'input_surface':input_surface_1})
        input_1, input_surface_1 = output_1, output_surface_1
        np.save('/scratch/08589/hvtran/Pangu-Weather/output_2023/'+os.path.basename(filei)+'_1hr_'+str(i),output_surface_1)
    
    input_6, input_surface_6 = input.copy(), input_surface.copy()
    print('6 hour')
    for i in range(4):
        output_6, output_surface_6 = ort_session_6.run(None, {'input':input_6, 'input_surface':input_surface_6})
        input_6, input_surface_6 = output_6, output_surface_6
        np.save('/scratch/08589/hvtran/Pangu-Weather/output_2023/'+os.path.basename(filei)+'_6hr_'+str(i),output_surface_6)
