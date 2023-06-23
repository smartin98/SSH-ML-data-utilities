# 2023-04-07 Scott Martin
# Revised and optimised data generation code for working on global SSH product

# this code defines a fixed grid of lon/lat points which are approximately equispaced by a distance of L km. These grid points will be the centers of the local patches used to create the global product. The code interpolates netcdf satellite datasets to .npy files containing data on local orthonormal projection grids for every day for the full record considered. These data will later be split for training-validation-testing purposes.

# variables to be interpolated:
    # CMEMS L3 SLA observations (un-gridded, time dependent)
    # CMEMS MDT (gridded, constant in t, lower res than target grid so INTERPOLATE) # CHECK TRUE AT ALL LATS
    # GEBCO Bathymetry (gridded, constant in t,higher res so BIN AVERAGE)
    # GHRSST MUR L4 SST IR+MW (gridded, time-dependent, higher res so BIN AVERAGE) #DEF TRUE AT ALL LATS
    # GHRSST L4 SST MW_OI (gridded, time-dependent, lower res so INTERPOLATE) #same as DUACS so check true for all lats
    # ERA 5 Winds (gridded, time-dependent, lower res so interpolate) #CHECK TRUE AT ALL LATS
    
    
# CODE UPDATED TO EXTRACT NRT OBSERVATIONS FROM 2021 & 2022, GO BACK AND FILL IN THE DATES THAT DIDN'T WORK IN THE DELAYED MODE DATA

import numpy as np
from numpy.random import randint
import pyproj
import scipy.spatial.transform 
import scipy.stats as stats
from scipy import interpolate
import matplotlib.path as mpltPath
import xarray as xr 
import time
from datetime import date, timedelta, datetime
import os
import multiprocessing
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from random import shuffle
import copy
from global_land_mask import globe

from global_data_utils import *

############ DEFINITIONS ######################

# Define the pyproj transformer objects used to transform coordinates between (lat,long,alt) and ECEF in both directions
transformer_ll2xyz = pyproj.Transformer.from_crs(
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        )
transformer_xyz2ll = pyproj.Transformer.from_crs(
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
        )

#################
# generate the lat/lon coords to center the local patches on:

L = 250 # spacing of reconstruction patches in km
R = 6378 # radius of earth in km
# lat-lon extent of global grid of reconstruction patches:
lat_min = -70
lat_max = 80
lon_min = -180
lon_max = 180
# create mesh of roughly equally spaced reconstruction patch centers:
dtheta = np.rad2deg(L/R)
lat = np.linspace(lat_min,lat_max,int((lat_max-lat_min)/dtheta))
coords = np.empty((int(1e5),2))
count = 0
dphis = np.zeros_like(lat)
for i in range(lat.shape[0]):
    dphi = np.rad2deg(L/(R*np.abs(np.cos(np.deg2rad(lat[i])))))
    lon_loop = lon_min + dphi/2
    while lon_loop<lon_max:
        coords[count,0] = lon_loop
        coords[count,1] = lat[i]
        count+=1
        lon_loop+=dphi
    if lon_loop-lon_max<=0.5*dphi:
        lon_loop = lon_max-dphi/2
        coords[count,0] = lon_loop
        coords[count,1] = lat[i]
        count+=1

coords = coords[:count,:]
# remove land points:
idx_ocean = []
for i in range(count):
    if ~globe.is_land(coords[i,1], coords[i,0]):
        idx_ocean.append(i)
ocean_coords = np.zeros((len(idx_ocean),2))
for i in range(len(idx_ocean)):
    ocean_coords[i,:] = coords[idx_ocean[i],:]
################

date_start = date(2010,1,1)
date_end = date(2022,12,31)
n_days = (date_end-date_start).days
n_centers = len(idx_ocean)
n = 128 # pixels in nxn local grids
L_x = 960e3 # size of local grid in m
L_y = 960e3 # size of local grid in m


data_bath = xr.open_dataset(os.path.expanduser('~')+'/dat1/aviso-data/gebco_bathymetry_4x_coarsened.nc')
data_duacs = xr.open_dataset('/dat1/smart1n/aviso-data/cnes_obs-sl_glo_phy-mdt_my_0.125deg_P20Y_1681506488705.nc') # CLS-CNES MDT


sst_hr_dir = '/dat1/smart1n/aviso-data/sst high res/'
sst_lr_dir = '/dat1/smart1n/aviso-data/sst mw l4/'
wind_dir = '/dat1/smart1n/aviso-data/wind/'

files_sst_lr = GetListOfFiles(sst_lr_dir)
files_sst_hr = GetListOfFiles(sst_hr_dir)
files_winds = GetListOfFiles(wind_dir)
files_u_winds = [f for f in files_winds if 'wind_u' in f]
files_v_winds = [f for f in files_winds if 'wind_v' in f]


def count_modified_files(directory):
    # Get the current time
    now = datetime.now()

    # Calculate the time threshold for the last 36 hours
    threshold = now - timedelta(hours=36)

    # Count the number of files modified within the last 24 hours
    count = 0
    modified_dates = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if modified_time >= threshold:
                count += 1
                # print(file)
                if 'ssh_tracks' in file:
                    modified_dates.append(date(int(file[-14:-10]),int(file[-9:-7]),int(file[-6:-4])))
    if len(modified_dates)==0:
        modified_dates.append(date(2020,12,31))
    return count, modified_dates


def save_files(center):
    
    # Perform the file-saving operation here
    # print(f"Saving files to directory: {directory}")
    save_dir = f'/dat1/smart1n/aviso-data/global training data/raw/{center}/'
    
    # n_modified_files, modified_dates = count_modified_files(save_dir)
    # print('start wait')
    # time.sleep(np.random.randint(10))
    # print('end wait')
    # files_saved = GetListOfFiles(save_dir, '.txt')
    # file_check = [f for f in files_saved if 'start_log_fixed_nrt_bug_fixmay24' in f]
    # n_saved = len(file_check)
    # print(n_saved)
    # n_saved=0
    
    # if n_saved==0:
        # with open(save_dir+'start_log_fixed_nrt_bug_fixmay24.txt', 'w') as f:
        #     f.write('started_nrt_fixed')
    # if np.max(modified_dates)<date(2022,12,29):  
    print(f'STARTING REGION {center}')
    lon0 = ocean_coords[center,0]
    lat0 = ocean_coords[center,1]
    # print('coords')
    coord_grid = grid_coords(data_bath, n, L_x, L_y, lon0, lat0)
    # print('bath')
    bath_grid = grid_bath(data_bath, n, L_x, L_y, lon0, lat0, coord_grid)

    np.save(save_dir+'coords.npy',coord_grid)
    np.save(save_dir+'bathymetry.npy',bath_grid)

    for t in range(n_days):
        # start = time.time() 
        date_loop = date_start + timedelta(days=t)

        if date_loop>date(2020,12,31):
            nrt=True
        else:
            nrt = False

        if nrt == False:
            satellites = ['alg','tpn','tp','s3b','s3a','j3','j2n','j2g','j2','j1n','j1g','j1','h2b','h2ag','h2a','g2','enn','en','e2','e1g','al','c2','c2n']
            sat_dir = '/dat1/smart1n/aviso-data/l3 sla data/'
        else:
            satellites = ['s3a','s3b','s6a','j3','j3n','al','c2n','h2b']
            sat_dir = '/dat1/smart1n/aviso-data/l3 sla data nrt/'
        # print(date_loop)

        # extract MDT
        if t==0:
            tri_mdt = mdt_delaunay(data_duacs, n, L_x, L_y, lon0, lat0)
            mdt = grid_mdt(data_duacs, 128, L_x, L_y, lon0, lat0,tri_mdt)

            np.save(save_dir+'mdt.npy',mdt)

        # extract along-track SSH obs:
        for s in range(len(satellites)):
            files_tracked = GetListOfFiles(sat_dir+satellites[s])
            if nrt==False:
                file = [f for f in files_tracked if f'_{date_loop}_'.replace('-','') in f]
            else:
                file = [f for f in files_tracked if f'_{date_loop}'.replace('-','') in f]
            if len(file)>0:
                data_tracked = xr.open_dataset(file[0])
                tracks = extract_tracked(data_tracked, L_x, L_y, lon0, lat0, transformer_ll2xyz, nrt)
                if tracks.shape[0]>5: # discard really short tracks
                    np.save(save_dir+'ssh_tracks_'+satellites[s]+f'_{date_loop}.npy',tracks)
            elif len(file)>1:
                raise Exception("len(sla file)>1")

        # grid low res SST:
        file_sst_lr = [f for f in files_sst_lr if f'/{date_loop}'.replace('-','') in f]
        if len(file_sst_lr)==1:
            data_sst_lr = xr.open_dataset(file_sst_lr[0])
            if t==0:
                sst_lr_delaunay_tri = sst_lr_delaunay(data_sst_lr, n, L_x, L_y, lon0, lat0)
            sst_lr = grid_sst_lr(data_sst_lr, n, L_x, L_y, lon0, lat0, sst_lr_delaunay_tri)
        elif len(file_sst_lr)>1:
            raise Exception("len(file_sst_lr)>1")
        else:
            sst_lr = np.zeros((n,n))

        np.save(save_dir+'sst_lr_'+f'{date_loop}.npy',sst_lr)

        # grid high res SST:
        file_sst_hr = [f for f in files_sst_hr if f'/{date_loop}'.replace('-','') in f]
        if len(file_sst_hr)==1:
            data_sst_hr = xr.open_dataset(file_sst_hr[0])
            sst_hr = grid_sst_hr(data_sst_hr, n, L_x, L_y, lon0, lat0, coord_grid)
        elif len(file_sst_hr)>1:
            print(file_sst_hr)
            raise Exception("len(file_sst_hr)>1") 
        else:
            sst_hr = np.zeros((n,n))

        np.save(save_dir+'sst_hr_'+f'{date_loop}.npy',sst_hr)

        # grid ERA 5 winds:
        wind_file_string = f'{date_loop.year}-{date_loop.month}'
        if len(wind_file_string)<7:
            wind_file_string = wind_file_string[:-1]+'0'+wind_file_string[-1]
        wind_file_u = [f for f in files_u_winds if wind_file_string in f]
        wind_file_v = [f for f in files_v_winds if wind_file_string in f]

        if len(wind_file_u)==1:
            data_wind_u = xr.open_dataset(wind_file_u[0])
            day = f'{date_loop.day}'
            if len(day)<2:
                day = '0'+day
            month = f'{date_loop.month}'
            if len(month)<2:
                month = '0'+month
            data_wind_u = data_wind_u.sel(time = f'{date_loop.year}-'+month+'-'+day)
            if t==0:
                wind_delaunay_tri = wind_delaunay(data_wind_u, n, L_x, L_y, lon0, lat0)
            if nrt==True:
                key_u = 'u10'
            else:
                key_u = 'uas'
            winds_u = grid_winds(data_wind_u, key_u, n, L_x, L_y, lon0, lat0, wind_delaunay_tri)
        elif len(wind_file_u)>1:
            raise Exception("len(wind_file_u)>1") 
        else:
            winds_u = np.zeros((n,n))

        if len(wind_file_v)==1:
            data_wind_v = xr.open_dataset(wind_file_v[0])
            day = f'{date_loop.day}'
            if len(day)<2:
                day = '0'+day
            month = f'{date_loop.month}'
            if len(month)<2:
                month = '0'+month
            data_wind_v = data_wind_v.sel(time = f'{date_loop.year}-'+month+'-'+day)
            if nrt==True:
                key_v = 'v10'
            else:
                key_v = 'vas'
            winds_v = grid_winds(data_wind_v, key_v, n, L_x, L_y, lon0, lat0, wind_delaunay_tri)
        elif len(wind_file_v)>1:
            raise Exception("len(wind_file_v)>1") 
        else:
            winds_v = np.zeros((n,n))

        np.save(save_dir+'winds_'+f'{date_loop}.npy',np.stack((winds_u,winds_v),axis=-1))

#                 end = time.time()

#                 print(end-start)
    print(f'FINISHED REGION {center}')
    # else:
    #     print(f'SKIPPED REGION {center}')


def worker(lock, centers):
    while True:
        # Acquire the lock to check and update the directories list
        with lock:
            if not centers:
                break  # No more directories to process

            center = centers.pop(0)  # Get the next directory
            print(f"Worker {multiprocessing.current_process().name} processing center: {center}")

        save_files(center)

def create_sublists(large_list, n):
    sublists = [[] for _ in range(n)]

    for i, element in enumerate(large_list):
        sublist_index = i % n
        sublists[sublist_index].append(element)

    return sublists

if __name__ == '__main__':
    centers = [i for i in range(5462,5586)]
    
    lock = multiprocessing.Lock()
    num_workers = 15
    centers_split = create_sublists(centers, num_workers)
    
    processes = []
    # centers_per_worker = len(centers) // num_workers
    # print(centers_per_worker)
    # worker_center_list = []
    # for w in range(num_workers-1):
    #     worker_center_list.append(centers[int(w*centers_per_worker):int((w+1)*centers_per_worker)])
    # worker_center_list.append(centers[int((num_workers-1)*centers_per_worker):])
    
    for i in range(num_workers):
        worker_centers = centers_split[i]
        print(f'worker{i}')
        print(worker_centers[0])
        print(worker_centers[-1])
        # centers = centers[centers_per_worker:]

        process = multiprocessing.Process(target=worker, args=(lock, worker_centers))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
    
    
