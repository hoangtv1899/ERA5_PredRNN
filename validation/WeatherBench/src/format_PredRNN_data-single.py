import os
import sys
import xarray as xr
import numpy as np
import xesmf as xe
import pandas as pd
from datetime import datetime, timedelta

def denormalize(array,key):
    norm_dict = {'u_wind':[-20,20],
                'v_wind':[-20,20],
                'temp':[210,305],
                'sea_press':[98000,105000],
                'precip':[-7,-2],
                'vapor':[-1,2.]}
    if key == 'wind':
        array = array * 20
    elif key == 'precip':
        array = 10**(array * (norm_dict[key][1] - norm_dict[key][0]) + norm_dict[key][0])
    else:
        array = array * (norm_dict[key][1] - norm_dict[key][0]) + norm_dict[key][0]
    return array

#create netCDF files
def write_netcdf(datas,key,var_dict,lon,lat,start_time,lead_time=None):
    if lead_time is not None:
        data_vars = {
            var_dict[key] : (["time","lead_time","lat","lon"], datas[key]),
        }
        coords = {
            "lon": lon,
            "lat": lat,
            "lead_time": lead_time,
            "time": start_time
        }
    else:
        data_vars = {
            var_dict[key] : (["time","lat","lon"], datas[key]),
        }
        coords = {
            "lon": lon,
            "lat": lat,
            "time": start_time
        }
    ds = xr.Dataset(data_vars = data_vars, coords = coords)
    return ds

#inputs
#in_dir = '/scratch/08589/hvtran/predrnn-pytorch/checkpoints/2012_predrnn_test/test_result/'
#out_dir = '/scratch/08589/hvtran/PredRNN_Sandy/'
in_dir = sys.argv[1]
out_dir = sys.argv[2]
start_date = datetime.strptime(sys.argv[3],'%Y-%m-%d-%H')
var_key = sys.argv[4]
#start_date = datetime(2012,10,21,12)
step = 24
list_batchs = sorted([int(name) for name in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, name))])
n_batchs = len(list_batchs)
curr_timestamp = datetime.now().strftime('%Y%m%d%H')
out_pd_file = out_dir+'raw/prediction_'+start_date.strftime('%Y%m%d%H')+'.'+var_key+'.'+curr_timestamp+'.nc'
out_gt_file = out_dir+'raw/truth_'+start_date.strftime('%Y%m%d%H')+'.'+var_key+'.'+curr_timestamp+'.nc'

#data parameters
var_dict = {'u_wind':'u10', 'wind':'v10', 'temp': 't2m', 'sea_press': 'sea_press', 'precip':'tp', 'vapor':'tcwv'}
var_order = {'wind':0, 'sea_press':1, 'precip':2}

lon = np.arange(0,360,0.25)
lat = np.arange(-89.75,90.1,0.25)[::-1]
start_time = pd.date_range(start_date, periods=n_batchs, freq=str(step)+'H')
start_time = np.array([(x - datetime(2001,1,1,0)).total_seconds()/3600. for x in start_time])

gt_final_arr = []
pd_final_arr = []

for batchi in list_batchs:
    #print(batchi)
    gt_arr = np.load(in_dir+str(batchi)+'/gt.npy')
    pd_arr = np.load(in_dir+str(batchi)+'/pd.npy')
    #appending
    if batchi < len(list_batchs):
        gt_final_arr.append(gt_arr[:,:step,:,:,:])
    else:
        gt_final_arr.append(gt_arr)
    pd_final_arr.append(pd_arr)

gt_final_arr = np.concatenate(gt_final_arr, axis=1)
pd_final_arr = np.concatenate(pd_final_arr, axis=0)

lead_time = np.arange(pd_arr.shape[1])
#denormalize
gt_u_arr = denormalize(gt_final_arr[0,:,:,:,var_order[var_key]],var_key)
pd_u_arr = denormalize(pd_final_arr[...,var_order[var_key]],var_key)

#write to netcdf files
pd_datas = {}
pd_datas[var_key] = pd_u_arr

pd_ds = write_netcdf(pd_datas,var_key,var_dict,lon,lat,start_time,lead_time)
pd_ds.time.attrs['units']='hours since 2001-01-01 00:00:00'
pd_ds.time.attrs['calendar']='standard'
pd_ds.time.encoding['units'] = 'hours since 2001-01-01 00:00:00'

if os.path.isfile(out_pd_file):
    os.remove(out_pd_file)

pd_ds.to_netcdf(out_pd_file)


gt_datas = {}
gt_datas[var_key] = gt_u_arr
#gt_datas['v_wind'] = gt_v_arr
#gt_datas['temp'] = gt_t_arr
#gt_datas['sea_press'] = gt_sp_arr
#gt_datas['precip'] = gt_p_arr
#gt_datas['vapor'] = gt_vp_arr

n_lead_time = gt_final_arr.shape[1]
start_time = pd.date_range(start_date, periods=n_lead_time, freq='H')
start_time = np.array([(x - datetime(2001,1,1,0)).total_seconds()/3600. for x in start_time])

gt_ds = write_netcdf(gt_datas,var_key,var_dict,lon,lat,start_time)
gt_ds.time.attrs['units']='hours since 2001-01-01 00:00:00'
gt_ds.time.attrs['calendar']='standard'
gt_ds.time.encoding['units'] = 'hours since 2001-01-01 00:00:00'

if os.path.isfile(out_gt_file):
    os.remove(out_gt_file)

gt_ds.to_netcdf(out_gt_file)

print(out_pd_file)
print(out_gt_file)

