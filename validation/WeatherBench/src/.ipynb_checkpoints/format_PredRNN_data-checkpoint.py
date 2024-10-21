import os
import sys
import xarray as xr
import numpy as np
import xesmf as xe
import pandas as pd
from datetime import datetime, timedelta

def denormalize(array):
    norm_dict = {'u_wind':[-20,20],
                'v_wind':[-20,20],
                'temp':[210,305],
                'sea_press':[98000,105000],
                'precip':[-7,-2],
                'vapor':[-1,2.]}
    array[:,:,:,:,0] = array[:,:,:,:,0] * 20
    array[:,:,:,:,1] = array[:,:,:,:,1] * 20
    array[:,:,:,:,2] = array[:,:,:,:,2] * (norm_dict['temp'][1] - norm_dict['temp'][0]) + norm_dict['temp'][0]
    array[:,:,:,:,3] = array[:,:,:,:,3] * (norm_dict['sea_press'][1] - norm_dict['sea_press'][0]) + norm_dict['sea_press'][0]
    array[:,:,:,:,4] = 10**(array[:,:,:,:,4] * (norm_dict['precip'][1] - norm_dict['precip'][0]) + norm_dict['precip'][0])
    array[:,:,:,:,5] = 10**(array[:,:,:,:,5] * (norm_dict['vapor'][1] - norm_dict['vapor'][0]) + norm_dict['vapor'][0])
    return array

#create netCDF files
def write_netcdf(datas,var_dict,lon,lat,start_time,lead_time=None):
    if lead_time is not None:
        data_vars = {
            var_dict['u_wind'] : (["time","lead_time","lat","lon"], datas['u_wind']),
            var_dict['v_wind'] :(["time","lead_time","lat","lon"], datas['v_wind']),
            var_dict['temp']: (["time","lead_time","lat","lon"], datas['temp']),
            var_dict['sea_press'] :(["time","lead_time","lat","lon"], datas['sea_press']),
            var_dict['precip'] : (["time","lead_time","lat","lon"], datas['precip']),
            var_dict['vapor'] : (["time","lead_time","lat","lon"], datas['vapor']),
        }
        coords = {
            "lon": lon,
            "lat": lat,
            "lead_time": lead_time,
            "time": start_time
        }
    else:
        data_vars = {
            var_dict['u_wind'] : (["time","lat","lon"], datas['u_wind']),
            var_dict['v_wind'] :(["time","lat","lon"], datas['v_wind']),
            var_dict['temp']: (["time","lat","lon"], datas['temp']),
            var_dict['sea_press'] :(["time","lat","lon"], datas['sea_press']),
            var_dict['precip'] : (["time","lat","lon"], datas['precip']),
            var_dict['vapor'] : (["time","lat","lon"], datas['vapor']),
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
#start_date = datetime(2012,10,21,12)
step = 24
n_lead_time = 240
list_batchs = sorted([name for name in os.listdir(in_dir) if os.path.isdir(os.path.join(in_dir, name))])
n_batchs = 1
curr_timestamp = datetime.now().strftime('%Y%m%d%H%M')
out_pd_file = out_dir+'/raw/prediction_'+start_date.strftime('%Y%m%d%H')+'_'+str(n_lead_time)+'.'+curr_timestamp+'.nc'
out_gt_file = out_dir+'/raw/truth_'+start_date.strftime('%Y%m%d%H')+'_'+str(n_lead_time)+'.'+curr_timestamp+'.nc'

#data parameters
var_dict = {'u_wind':'u10', 'v_wind':'v10', 'temp': 't2m', 'sea_press': 'sea_press', 'precip':'tp', 'vapor':'tcwv'}
lon = np.arange(0,360,0.25)
lat = np.arange(-89.75,90.1,0.25)[::-1]
start_time = pd.date_range(start_date, periods=n_batchs, freq=str(step)+'H')
start_time = np.array([(x - datetime(2001,1,1,0)).total_seconds()/3600. for x in start_time])
lead_time = np.arange(n_lead_time).astype(np.float64)
"""
# loop through the result files
gt_u_arr = np.zeros((n_batchs*step + n_lead_time, len(lat), len(lon)))
gt_v_arr = np.zeros((n_batchs*step + n_lead_time, len(lat), len(lon)))
gt_t_arr = np.zeros((n_batchs*step + n_lead_time, len(lat), len(lon)))
gt_sp_arr = np.zeros((n_batchs*step + n_lead_time, len(lat), len(lon)))
gt_p_arr = np.zeros((n_batchs*step + n_lead_time, len(lat), len(lon)))
gt_vp_arr = np.zeros((n_batchs*step + n_lead_time, len(lat), len(lon)))

pd_u_arr = np.zeros((n_batchs,n_lead_time, len(lat), len(lon)))
pd_v_arr = np.zeros((n_batchs,n_lead_time, len(lat), len(lon)))
pd_t_arr = np.zeros((n_batchs,n_lead_time, len(lat), len(lon)))
pd_sp_arr = np.zeros((n_batchs,n_lead_time, len(lat), len(lon)))
pd_p_arr = np.zeros((n_batchs,n_lead_time, len(lat), len(lon)))
pd_vp_arr = np.zeros((n_batchs,n_lead_time, len(lat), len(lon)))
"""
gt_arr = np.load(in_dir + '/gt.npy')
gt_arr = gt_arr[:,-1*n_lead_time:,:,:,:]
pd_arr = np.load(in_dir + '/pd.npy')
pd_arr = pd_arr * 2
gt_arr[...,2] = gt_arr[...,2] + 0.5
gt_arr[...,5] = gt_arr[...,5] + 0.5
pd_arr[...,2] = pd_arr[...,2] + 0.5
pd_arr[...,5] = pd_arr[...,5] + 0.5
#denormalize
gt_arr = denormalize(gt_arr)
pd_arr = denormalize(pd_arr)
#append to final arrays
gt_u_arr = gt_arr[0,:,:,:,0]
gt_v_arr = gt_arr[0,:,:,:,1]
gt_t_arr = gt_arr[0,:,:,:,2]
gt_sp_arr = gt_arr[0,:,:,:,3]
gt_p_arr = gt_arr[0,:,:,:,4]
gt_vp_arr = gt_arr[0,:,:,:,5]

pd_u_arr = pd_arr[0,:,:,:,0][np.newaxis,...]
pd_v_arr = pd_arr[0,:,:,:,1][np.newaxis,...]
pd_t_arr = pd_arr[0,:,:,:,2][np.newaxis,...]
pd_sp_arr = pd_arr[0,:,:,:,3][np.newaxis,...]
pd_p_arr = pd_arr[0,:,:,:,4][np.newaxis,...]
pd_vp_arr = pd_arr[0,:,:,:,5][np.newaxis,...]

#write to netcdf files
pd_datas = {}
pd_datas['u_wind'] = pd_u_arr
pd_datas['v_wind'] = pd_v_arr
pd_datas['temp'] = pd_t_arr
pd_datas['sea_press'] = pd_sp_arr
pd_datas['precip'] = pd_p_arr
pd_datas['vapor'] = pd_vp_arr

pd_ds = write_netcdf(pd_datas,var_dict,lon,lat,start_time,lead_time)
pd_ds.time.attrs['units']='hours since 2001-01-01 00:00:00'
pd_ds.time.attrs['calendar']='standard'
pd_ds.time.encoding['units'] = 'hours since 2001-01-01 00:00:00'
pd_ds.to_netcdf(out_pd_file)

gt_datas = {}
gt_datas['u_wind'] = gt_u_arr
gt_datas['v_wind'] = gt_v_arr
gt_datas['temp'] = gt_t_arr
gt_datas['sea_press'] = gt_sp_arr
gt_datas['precip'] = gt_p_arr
gt_datas['vapor'] = gt_vp_arr

start_time = pd.date_range(start_date, periods=n_lead_time, freq='H')
start_time = np.array([(x - datetime(2001,1,1,0)).total_seconds()/3600. for x in start_time])
gt_ds = write_netcdf(gt_datas,var_dict,lon,lat,start_time)
gt_ds.time.attrs['units']='hours since 2001-01-01 00:00:00'
gt_ds.time.attrs['calendar']='standard'
gt_ds.time.encoding['units'] = 'hours since 2001-01-01 00:00:00'
gt_ds.to_netcdf(out_gt_file)



