# 2023-04-07 Scott Martin
# Revised and optimised data generation code for working on global SSH product

# this code defines a fixed grid of lon/lat points which are approximately equispaced by a distance of L km. These grid points will be the centers of the local patches used to create the global product. The code interpolates netcdf satellite datasets to .npy files containing data on local orthonormal projection grids for every day for the full record considered. These data will later be split for training-validation-testing purposes.

# variables to be interpolated:
    # CMEMS L3 SLA observations (un-gridded, time dependent)
    # CMEMS MDT (gridded, constant in t, lower res than target grid so INTERPOLATE)
    # GHRSST MUR L4 SST IR+MW (gridded, time-dependent, higher res so BIN AVERAGE) 
    # GHRSST L4 SST MW_OI (gridded, time-dependent, lower res so INTERPOLATE) 

## code first generates a grid of lat/lon points on which to center training examples. Create subdirectories named with the index (0 to n) for each central point on that grid. Within those subdirectories it'll save a .npy file for each day for each available data type along with lat/lon coordinates for the local grid.
    
import numpy as np
from numpy.random import randint
import pyproj
import scipy.spatial.transform 
import scipy.stats as stats
from scipy import interpolate
import matplotlib.path as mpltPath
import xarray as xr 
import time
from datetime import date, timedelta
import os
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
lat_max = 65
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
date_end = date(2020,12,31)
n_days = (date_end-date_start).days
n_centers = len(idx_ocean)
n = 128 # pixels in nxn local grids
L_x = 960e3 # size of local grid in m
L_y = 960e3 # size of local grid in m


data_bath = xr.open_dataset(os.path.expanduser('~')+'/dat1/aviso-data/gebco_bathymetry_4x_coarsened.nc')
data_duacs = xr.open_dataset('/dat1/smart1n/aviso-data/cnes_obs-sl_glo_phy-mdt_my_0.125deg_P20Y_1681506488705.nc') # CLS-CNES MDT

satellites = ['alg','tpn','tp','s3b','s3a','j3','j2n','j2g','j2','j1n','j1g','j1','h2b','h2ag','h2a','g2','enn','en','e2','e1g','al','c2','c2n']
nrt = False # change to true if using SLA data from the last 2 years where CMEMS near real-time product required
sat_dir = '/dat1/smart1n/aviso-data/l3 sla data/'
sst_hr_dir = '/dat1/smart1n/aviso-data/sst high res/'
sst_lr_dir = '/dat1/smart1n/aviso-data/sst mw l4/'

files_sst_lr = GetListOfFiles(sst_lr_dir)
files_sst_hr = GetListOfFiles(sst_hr_dir)


for center in range(n_centers):
    save_dir = f'/dat1/smart1n/aviso-data/global training data/raw/{center}/'

    print(f'STARTING REGION {center}')
    lon0 = ocean_coords[center,0]
    lat0 = ocean_coords[center,1]
    print('coords')
    coord_grid = grid_coords(data_bath, n, L_x, L_y, lon0, lat0)
    print('bath')
    bath_grid = grid_bath(data_bath, n, L_x, L_y, lon0, lat0, coord_grid)

    np.save(save_dir+'coords.npy',coord_grid)
    np.save(save_dir+'bathymetry.npy',bath_grid)
    
    for t in range(n_days):
        
        start = time.time()
        
        date_loop = date_start + timedelta(days=t)
        print(date_loop)
        
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
        
        end = time.time()
        
        print(end-start)
    print(f'FINISHED REGION {center}')
        

            
        
                
                
