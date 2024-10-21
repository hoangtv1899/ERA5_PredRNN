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

grib_file = 'ERA5_validation.grib'
ds= pygrib.open(grib_file)

curr_date = datetime(2023,1,1,0)
u_wind_arr = []
v_wind_arr = []
sea_press_arr = []
temp_2m_arr = []

for g in ds:
    datej = datetime.strptime(str(g.date)+str(g.hour), '%Y%m%d%H')
    if datej != curr_date:
        print(curr_date)
        print(len(u_wind_arr), len(v_wind_arr), len(sea_press_arr), len(temp_2m_arr))
        final_arr = np.vstack([u_wind_arr[0][np.newaxis,...],
                               v_wind_arr[0][np.newaxis,...],
                               sea_press_arr[0][np.newaxis,...],
                               temp_2m_arr[0][np.newaxis,...],
                              ])
        np.save('/pscratch/sd/h/hvtran/ERA5/2023/'+curr_date.strftime('%Y%m%d%H'), final_arr)
        curr_date = datej
        u_wind_arr = []
        v_wind_arr = []
        sea_press_arr = []
        temp_2m_arr = []
    if '10 metre U wind component' == str(g.name):
        u_wind_arr.append(g.values.astype('float32'))
    elif '10 metre V wind component' == str(g.name):
        v_wind_arr.append(g.values.astype('float32'))
    elif 'Mean sea level pressure' == str(g.name):
        sea_press_arr.append(g.values.astype('float32'))
    elif '2 metre temperature' == str(g.name):
        temp_2m_arr.append(g.values.astype('float32'))


