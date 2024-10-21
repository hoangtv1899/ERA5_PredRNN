import os
from time import sleep
#from osgeo import gdal
import numpy as np
from glob import glob
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta
import pygrib
import sys

date_info = sys.argv[1]
file_no = int(sys.argv[2])

ds = xr.open_dataset('dataset_source-era5_date-2022-01-01_res-0.25_levels-13_steps-01.nc')

surface_grib_file = '/scratch/08589/hvtran/new_ERA5/surface_2023.grib'

list_upper_gribs = ['upper_2023_1.grib',
                   'upper_2023_2.grib',
                   'upper_2023_3.grib',
                   'upper_2023_4.grib',
                   'upper_2023_5.grib',
                   'upper_2023_6.grib',
                   'upper_2023_7.grib',
                   'upper_2023_8.grib',
                   'upper_2023_9.grib',
                   'upper_2023_10.grib',
                   'upper_2023_11.grib',
                   'upper_2023_12.grib']


datei = datetime.strptime(date_info,"%Y%m%d")
datei_prev = datei-timedelta(hours=6)
datei_prev_prev = datei-timedelta(hours=18)

outfile = '/scratch/08589/hvtran/GC/input/input_surface_'+datei.strftime('%Y-%m-%d-%H')+'.nc'

surface_ds = pygrib.open(surface_grib_file)
upper_ds = pygrib.open('/scratch/08589/hvtran/new_ERA5/'+list_upper_gribs[file_no])

u_wind_arr = []
v_wind_arr = []
sea_press_arr = []
temp_2m_arr = []
toa_arr = []
precip_arr = []

for g in surface_ds:
    datej = datetime.strptime(str(g.date)+str(g.hour), '%Y%m%d%H')
    if datej in [datei, datei_prev,datei_prev_prev]:
        #print(datej)
        if '10 metre U wind component' == str(g.name):
            u_wind_arr.append(g.values.astype('float32'))
        elif '10 metre V wind component' == str(g.name):
            v_wind_arr.append(g.values.astype('float32'))
        elif 'Mean sea level pressure' == str(g.name):
            sea_press_arr.append(g.values.astype('float32'))
        elif '2 metre temperature' == str(g.name):
            temp_2m_arr.append(g.values.astype('float32'))
        elif 'TOA incident solar radiation' == str(g.name):
            toa_arr.append(g.values.astype('float32'))
        elif 'Total precipitation' == str(g.name):
            precip_arr.append(g.values.astype('float32'))

temp_2m_arr = np.vstack([x[np.newaxis,...] for x in temp_2m_arr])[np.newaxis,...]
sea_press_arr = np.vstack([x[np.newaxis,...] for x in sea_press_arr])[np.newaxis,...]
v_wind_arr = np.vstack([x[np.newaxis,...] for x in v_wind_arr])[np.newaxis,...]
u_wind_arr = np.vstack([x[np.newaxis,...] for x in u_wind_arr])[np.newaxis,...]
precip_arr = np.vstack([x[np.newaxis,...] for x in precip_arr])[np.newaxis,...]
toa_arr = np.vstack([x[np.newaxis,...] for x in toa_arr])[np.newaxis,...]

geopot_arr = {}
SH_arr = {}
T_arr = {}
U_arr = {}
V_arr = {}
veloc_arr = {}

for g in upper_ds:
    datej = datetime.strptime(str(g.date)+str(g.hour), '%Y%m%d%H')
    if datej in [datei, datei_prev]:
        #print(datej)
        if 'Geopotential' == str(g.name):
            if g.level in geopot_arr.keys():
                geopot_arr[g.level].append(g.values.astype('float32'))
            else:
                geopot_arr[g.level] = [g.values.astype('float32')]
        elif 'Specific humidity' == str(g.name):
            if g.level in SH_arr.keys():
                SH_arr[g.level].append(g.values.astype('float32'))
            else:
                SH_arr[g.level] = [g.values.astype('float32')]
        elif 'Temperature' == str(g.name):
            if g.level in T_arr.keys():
                T_arr[g.level].append(g.values.astype('float32'))
            else:
                T_arr[g.level] = [g.values.astype('float32')]
        elif 'U component of wind' == str(g.name):
            if g.level in U_arr.keys():
                U_arr[g.level].append(g.values.astype('float32'))
            else:
                U_arr[g.level] = [g.values.astype('float32')]
        elif 'V component of wind' == str(g.name):
            if g.level in V_arr.keys():
                V_arr[g.level].append(g.values.astype('float32'))
            else:
                V_arr[g.level] = [g.values.astype('float32')]
        elif 'Vertical velocity' == str(g.name):
            if g.level in veloc_arr.keys():
                veloc_arr[g.level].append(g.values.astype('float32'))
            else:
                veloc_arr[g.level] = [g.values.astype('float32')]

if len(veloc_arr) == 0:
    print('error')

upper_arr = np.zeros((2,6,13,721,1440))

for i in range(2):
    for j, leveli in enumerate([1000, 925, 850, 700, 600,
                                500, 400, 300, 250, 200,
                                150, 100, 50]):
        upper_arr[i,1,j,...] = geopot_arr[leveli][i]
        upper_arr[i,5,j,...] = SH_arr[leveli][i]
        upper_arr[i,0,j,...] = T_arr[leveli][i]
        upper_arr[i,2,j,...] = U_arr[leveli][i]
        upper_arr[i,3,j,...] = V_arr[leveli][i]
        upper_arr[i,4,j,...] = veloc_arr[leveli][i]

coords = {
        "lon": ds.lon,
        "lat": ds.lat,
        "level": ds.level,
        "time": ds.time[:2],
        "datetime": np.array([datei_prev,datei])}
dims = ["lon", "lat", "level", "time", "batch"]

var_dict0 = ["batch", "time", "lat", "lon" ]
var_dict1 = ["batch", "time", "level", "lat", "lon" ]

data_vars = {
            "geopotential_at_surface" : (["lat", "lon"], np.array(ds["geopotential_at_surface"])),
            "land_sea_mask" : (["lat", "lon"], np.array(ds["land_sea_mask"])),
    "2m_temperature" : (var_dict0, temp_2m_arr[:,:,::-1,:]),
    "mean_sea_level_pressure" : (var_dict0, sea_press_arr[:,:,::-1,:]),
    "10m_v_component_of_wind" : (var_dict0, v_wind_arr[:,:,::-1,:]),
    "10m_u_component_of_wind" : (var_dict0, u_wind_arr[:,:,::-1,:]),
    "total_precipitation_6hr" : (var_dict0, precip_arr[:,:,::-1,:]),
    "toa_incident_solar_radiation" : (var_dict0, toa_arr[:,:,::-1,:]),
    "temperature" : (var_dict1, upper_arr[:,0,::-1,::-1,:][np.newaxis,...]),
    "geopotential" : (var_dict1, upper_arr[:,1,::-1,::-1,:][np.newaxis,...]),
    "u_component_of_wind" : (var_dict1, upper_arr[:,2,::-1,::-1,:][np.newaxis,...]),
    "v_component_of_wind" : (var_dict1, upper_arr[:,3,::-1,::-1,:][np.newaxis,...]),
    "vertical_velocity" : (var_dict1, upper_arr[:,4,::-1,::-1,:][np.newaxis,...]),
    "specific_humidity" : (var_dict1, upper_arr[:,5,::-1,::-1,:][np.newaxis,...]),
            }

out_ds = xr.Dataset(data_vars = data_vars, coords = coords)
out_ds.to_netcdf(outfile)

print('done')
