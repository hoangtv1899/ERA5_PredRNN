import os
import sys
import xarray as xr
import numpy as np
import xesmf as xe
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns
from collections import OrderedDict

from score import *

import pickle
def to_pickle(obj, fn):
    with open(fn, 'wb') as f:
        pickle.dump(obj, f)
def read_pickle(fn):
    with open(fn, 'rb') as f:
        return pickle.load(f)

file_loc = sys.argv[1]
start_date = datetime.strptime(sys.argv[2],'%Y-%m-%d-%H')
skip_time = int(sys.argv[3])
out_dir = sys.argv[4]

with open(file_loc,'r') as fi:
    content = fi.read()

pred_files = []
valid_files = []
for linei in content.split('\n'):
    if 'prediction' in linei:
        pred_files.append(linei)
    elif 'truth' in linei:
        valid_files.append(linei)

pred_list = []
for pred_filei in pred_files:
    temp_pred_ds = xr.open_dataset(pred_filei)
    if 'precip' in pred_filei:
        pred_ds = temp_pred_ds.cumsum(dim=['lead_time'])
    pred_list.append(temp_pred_ds)

valid_list = []
for valid_filei in valid_files:
    temp_valid_ds = xr.open_dataset(valid_filei)
    if 'precip' in valid_filei:
        temp_valid_ds = temp_valid_ds.cumsum(dim=['time'])
    valid_list.append(temp_valid_ds)

pred_ds = xr.merge(pred_list)
valid_ds = xr.merge(valid_list)

pred_ds = pred_ds.isel(lead_time=slice(0, None, skip_time))
valid_ds = valid_ds.isel(time=slice(0, None, skip_time))

valid_ds = valid_ds.isel(time=slice(24,None,1))

pred_ds = pred_ds.isel(time=slice(0,56,1))
pred_ds['time'] = valid_ds.isel(time=slice(0,None,24)).time[:-2]

print(valid_ds, pred_ds)

func = compute_weighted_rmse
rmse = OrderedDict({
    'PredRNN': evaluate_iterative_forecast(pred_ds, valid_ds, func).load(),
})

to_pickle(rmse, f'{out_dir}rmse.pkl')

func = compute_weighted_acc
acc = OrderedDict({
    'PredRNN': evaluate_iterative_forecast(pred_ds, valid_ds, func).load(),
})

to_pickle(acc, f'{out_dir}acc.pkl')

func = compute_weighted_mae
mae = OrderedDict({
    'PredRNN': evaluate_iterative_forecast(pred_ds, valid_ds, func).load(),
})

to_pickle(mae, f'{out_dir}mae.pkl')

