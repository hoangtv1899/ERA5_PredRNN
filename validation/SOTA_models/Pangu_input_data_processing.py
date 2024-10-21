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

####Processing surface inputs
def all_lists_not_empty(list_of_lists):
    return all(lst for lst in list_of_lists)

surface_grib_file = '/scratch/08589/hvtran/new_ERA5/surface_2023.grib'
data_ds2 = pygrib.open(surface_grib_file)

u_wind_arr = []
v_wind_arr = []
sea_press_arr = []
temp_2m_arr = []
result_list = [u_wind_arr,v_wind_arr,sea_press_arr,temp_2m_arr]

existing_dates = []


for ii,g in enumerate(data_ds2):
    if g.hour != 0:
        continue    
    datei = datetime.strptime(str(g.date)+str(g.hour), '%Y%m%d%H')
    print(datei)
    outfile = '/scratch/08589/hvtran/Pangu-Weather/input_2023/input_surface_'+datei.strftime('%Y-%m-%d-%H')+'.npy'
    if datei.strftime('%Y-%m-%d-%H') not in existing_dates:
        u_wind_arr = []
        v_wind_arr = []
        sea_press_arr = []
        temp_2m_arr = []
        
        existing_dates.append(datei.strftime('%Y-%m-%d-%H'))
        
        if '10 metre U wind component' == str(g.name):
            u_wind_arr.append(g.values.astype('float32'))
        elif '10 metre V wind component' == str(g.name):
            v_wind_arr.append(g.values.astype('float32'))
        elif 'Mean sea level pressure' == str(g.name):
            sea_press_arr.append(g.values.astype('float32'))
        elif '2 metre temperature' == str(g.name):
            temp_2m_arr.append(g.values.astype('float32'))
        result_list = [u_wind_arr,v_wind_arr,sea_press_arr,temp_2m_arr]
    else:
        if '10 metre U wind component' == str(g.name):
            u_wind_arr.append(g.values.astype('float32'))
        elif '10 metre V wind component' == str(g.name):
            v_wind_arr.append(g.values.astype('float32'))
        elif 'Mean sea level pressure' == str(g.name):
            sea_press_arr.append(g.values.astype('float32'))
        elif '2 metre temperature' == str(g.name):
            temp_2m_arr.append(g.values.astype('float32'))
        result_list = [u_wind_arr,v_wind_arr,sea_press_arr,temp_2m_arr]
        
        if all_lists_not_empty(result_list):
            surface_arr = np.vstack([
                                sea_press_arr[0][np.newaxis,...],
                                u_wind_arr[0][np.newaxis,...],
                                v_wind_arr[0][np.newaxis,...],
                                temp_2m_arr[0][np.newaxis,...]
                                ])
            print('save '+outfile)
            np.save(outfile, surface_arr)


####Processing upper inputs
def all_dict_equal_len(list_of_dicts):
    return all(len(dct.keys())==13 for dct in list_of_dicts)

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

geopot_arr = {}
SH_arr = {}
T_arr = {}
U_arr = {}
V_arr = {}
existing_dates = []
result_list = [geopot_arr,SH_arr,T_arr,U_arr,V_arr]

for filei in list_upper_gribs:
    temp_ds = pygrib.open('/scratch/08589/hvtran/new_ERA5/'+filei)
    for g in temp_ds:
        if g.hour != 0:
            continue    
        datei = datetime.strptime(str(g.date)+str(g.hour), '%Y%m%d%H')
        print(datei)
        outfile = '/scratch/08589/hvtran/Pangu-Weather/input_2023/input_upper_'+datei.strftime('%Y-%m-%d-%H')+'.npy'
        if datei.strftime('%Y-%m-%d-%H') not in existing_dates:
            geopot_arr = {}
            SH_arr = {}
            T_arr = {}
            U_arr = {}
            V_arr = {}

            existing_dates.append(datei.strftime('%Y-%m-%d-%H'))

            if 'Geopotential' == str(g.name):
                if g.level not in geopot_arr.keys():
                    geopot_arr[g.level] = [g.values.astype('float32')]
            elif 'Specific humidity' == str(g.name):
                if g.level not in SH_arr.keys():
                    SH_arr[g.level] = [g.values.astype('float32')]
            elif 'Temperature' == str(g.name):
                if g.level not in T_arr.keys():
                    T_arr[g.level] = [g.values.astype('float32')]
            elif 'U component of wind' == str(g.name):
                if g.level not in U_arr.keys():
                    U_arr[g.level] = [g.values.astype('float32')]
            elif 'V component of wind' == str(g.name):
                if g.level not in V_arr.keys():
                    V_arr[g.level] = [g.values.astype('float32')]
            else:
                continue
                    
        else:
            if 'Geopotential' == str(g.name):
                if g.level not in geopot_arr.keys():
                    geopot_arr[g.level] = [g.values.astype('float32')]
            elif 'Specific humidity' == str(g.name):
                if g.level not in SH_arr.keys():
                    SH_arr[g.level] = [g.values.astype('float32')]
            elif 'Temperature' == str(g.name):
                if g.level not in T_arr.keys():
                    T_arr[g.level] = [g.values.astype('float32')]
            elif 'U component of wind' == str(g.name):
                if g.level not in U_arr.keys():
                    U_arr[g.level] = [g.values.astype('float32')]
            elif 'V component of wind' == str(g.name):
                if g.level not in V_arr.keys():
                    V_arr[g.level] = [g.values.astype('float32')]
            else:
                continue
            
        result_list = [geopot_arr,SH_arr,T_arr,U_arr,V_arr]
        if all_dict_equal_len(result_list):
            upper_arr = np.zeros((5,13,721,1440))
            for j, leveli in enumerate([1000, 925, 850, 700, 600,
                            500, 400, 300, 250, 200,
                            150, 100, 50]):
                upper_arr[0,j,...] = geopot_arr[leveli][0]
                upper_arr[1,j,...] = SH_arr[leveli][0]
                upper_arr[2,j,...] = T_arr[leveli][0]
                upper_arr[3,j,...] = U_arr[leveli][0]
                upper_arr[4,j,...] = V_arr[leveli][0]
            print('save '+outfile)
            np.save(outfile, upper_arr)

