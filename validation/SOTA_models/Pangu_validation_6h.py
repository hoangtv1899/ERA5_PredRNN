import os
import xarray as xr
import numpy as np
#import xesmf as xe
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns
from glob import glob
from collections import OrderedDict
from calendar import monthrange
import pygrib
import sys

from score import *


def get_fc_array(dt):
    ########default vars
    lat = np.linspace(-np.pi/2, np.pi/2, 721)
    lat_int = np.linspace(-np.pi/2, np.pi/2, 720)
    dx = np.pi/720
    lat_diff = np.reshape(lat_int-lat[:-1], (-1,1)).astype('float32')
    basepath = '/pscratch/sd/h/hvtran/Pangu/'
    basename = 'input_surface_'
    postfix='.npy'
    ###############
    
    di = dt
    #print(di)
    temp_array = []
    for j in range(4):
        f0i = basename+di.strftime('%Y-%m-%d-%H')+'.npy_6hr_'+str(j)+'.npy'
        f0i = basepath+f0i
        if not os.path.isfile(f0i):
            return None
        time_step_arr = np.load(f0i)[[1,2,0,3],...]
        temp_array.append(time_step_arr[np.newaxis,...])
    return np.vstack(temp_array)


#get ERA5 data
def get_era5(dt):
    infile = dt.strftime('%Y%m%d%H')+'.npy'

    output_path = '/pscratch/sd/h/hvtran/ERA5/2023/'
    
    valid_arr = np.load(output_path+infile)
    return valid_arr

### create netCDF files
def np2xr(datas,var_dict,lon,lat,time_period,variables):
    data_vars = {}
    temp_var_dict = ["time","lat","lon"]
    for ii, vari in enumerate(variables):
        data_vars[vari] = (temp_var_dict,datas[:,ii,:,:])
    
    coords = {
        "time": time_period,
        "lon": lon,
        "lat": lat,
        
    }
    ds = xr.Dataset(data_vars = data_vars, coords = coords)
    return ds



def cal_stats(which_month, offset, existing_acc_ds=None, existing_rmse_ds=None):
    start_datei = datetime(2023,which_month,1,0)
    lon = np.arange(0,360,0.25)
    lat = np.arange(-89.75,90.5,0.25)[::-1]
    var_dict = ["time", "variables", "lat", "lon" ]
    variables = ['u10', 'v10', 'mslp', 't2m']
    time_step = 6
    rmse_arrs = []
    acc_arrs = []

    curr_date = start_datei + timedelta(hours=offset*24)
    pred_arr = get_fc_array(curr_date)
    #pred_arr = pred_arr[:,:,::-1,:]
    valid_arr = []
    for j in np.arange(1,5):
        #print(curr_date+timedelta(hours=int(j)*6))
        temp_valid = get_era5(curr_date+timedelta(hours=int(j)*6))
        valid_arr.append(temp_valid[np.newaxis,...])

    valid_arr = np.vstack(valid_arr)

    #print(valid_arr.shape)
    datetime_arrays = [curr_date + timedelta(hours=int(x)) for x in np.arange(6,25,time_step)]
    time_ds = np.array([(x - datetime(2001,1,1,0)).total_seconds()/3600. for x in datetime_arrays])
    sim_xr = np2xr(pred_arr,var_dict,lon,lat,time_ds,variables)
    obs_xr = np2xr(valid_arr,var_dict,lon,lat,time_ds,variables)

    curr_dt_rmse = []
    curr_dt_acc = []
    for timei in time_ds:
        temp_rmse = compute_weighted_rmse(sim_xr.sel(time=slice(timei,timei)), obs_xr)
        temp_acc = compute_weighted_acc(sim_xr.sel(time=slice(timei,timei)), obs_xr)
        curr_dt_rmse.append(temp_rmse.expand_dims(time=[curr_date]))
        curr_dt_acc.append(temp_acc.expand_dims(time=[curr_date]))

    rmse_arrs.append(xr.concat(curr_dt_rmse, 'lead_time'))
    acc_arrs.append(xr.concat(curr_dt_acc, 'lead_time'))
    
    return xr.concat(rmse_arrs, 'time'), xr.concat(acc_arrs, 'time')



which_month = int(sys.argv[1])

if os.path.isfile('/global/homes/h/hvtran/haoli/ERA5_PredRNN/validation/results/Pangu_6h_2023_acc_'+str(which_month)+'.nc'):
    existing_acc_ds = xr.open_dataset('/global/homes/h/hvtran/haoli/ERA5_PredRNN/validation/results/Pangu_6h_2023_acc_'+str(which_month)+'.nc')
    existing_rmse_ds = xr.open_dataset('/global/homes/h/hvtran/haoli/ERA5_PredRNN/validation/results/Pangu_6h_2023_rmse_'+str(which_month)+'.nc')
    os.remove('/global/homes/h/hvtran/haoli/ERA5_PredRNN/validation/results/Pangu_6h_2023_acc_'+str(which_month)+'.nc')
    os.remove('/global/homes/h/hvtran/haoli/ERA5_PredRNN/validation/results/Pangu_6h_2023_rmse_'+str(which_month)+'.nc')
else:
    existing_acc_ds, existing_rmse_ds = None, None

#####loop through the files and calculate the statistics
part_acc = []
part_rmse = []

for i in range(1,monthrange(2023,which_month)[1]):
    rmsei, acci = cal_stats(which_month, i,existing_acc_ds, existing_rmse_ds)
    if rmsei is not None:
        part_acc.append(acci)
        part_rmse.append(rmsei)
        temp_acc = xr.concat(part_acc, 'time')
        temp_acc.to_netcdf('/global/homes/h/hvtran/haoli/ERA5_PredRNN/validation/results/Pangu_6h_2023_acc_'+str(which_month)+'.nc', mode='w')

        temp_rmse = xr.concat(part_rmse, 'time')
        temp_rmse.to_netcdf('/global/homes/h/hvtran/haoli/ERA5_PredRNN/validation/results/Pangu_6h_2023_rmse_'+str(which_month)+'.nc', mode='w')
        print(i)

final_acc = xr.concat(part_acc, 'time')
final_acc.to_netcdf('/global/homes/h/hvtran/haoli/ERA5_PredRNN/validation/results/Pangu_6h_2023_acc_'+str(which_month)+'.nc', mode='w')

final_rmse = xr.concat(part_rmse, 'time')
final_rmse.to_netcdf('/global/homes/h/hvtran/haoli/ERA5_PredRNN/validation/results/Pangu_6h_2023_rmse_'+str(which_month)+'.nc', mode='w')
