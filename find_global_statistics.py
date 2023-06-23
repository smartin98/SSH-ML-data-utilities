import numpy as np
import os
import datetime
import random

# function to list all files within a directory including within any subdirectories
def GetListOfFiles(dirName, ext = '.nc'):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + GetListOfFiles(fullPath)
        else:
            if fullPath.endswith(ext):
                allFiles.append(fullPath)               
    return allFiles

N_samples =50000

start_date = datetime.date(2010,1,1)
end_date = datetime.date(2022,12,31)
n_days = (end_date-start_date).days + 1
val_dates = []
for t in range(73-30):
    val_dates.append(datetime.date(2010,1,1)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2011,1,1)+datetime.timedelta(days = 73)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2012,1,1)+datetime.timedelta(days = 2*73)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2013,1,1)+datetime.timedelta(days = 3*73)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2014,1,1)+datetime.timedelta(days = 4*73)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2016,1,1)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2017,1,1)+datetime.timedelta(days = 73)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2018,1,1)+datetime.timedelta(days = 2*73)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2020,1,1)+datetime.timedelta(days = 3*73)+datetime.timedelta(days = 15+t))
    val_dates.append(datetime.date(2021,1,1)+datetime.timedelta(days = 4*73)+datetime.timedelta(days = 15+t))
test_dates = []
for t in range(365):
    test_dates.append(datetime.date(2019,1,1)+datetime.timedelta(days=t))

train_dates = []
for t in range(n_days-15):
    check_date = start_date+ datetime.timedelta(days=t)
    diffs_val = [np.abs((check_date-date).days) for date in val_dates]
    diffs_test = [np.abs((check_date-date).days) for date in test_dates]
    
    if (np.min(diffs_val)>=30) and (np.min(diffs_test)>=30):
        train_dates.append(check_date)

# sla_mean = np.zeros(N_samples)
# sla_std = np.zeros(N_samples)
# dsla_std = np.zeros(N_samples)
# d2sla_std = np.zeros(N_samples)

sla_sum = np.zeros(N_samples)
sla2_sum = np.zeros(N_samples)
n_samples = np.zeros(N_samples)

sst_hr_sum = np.zeros(N_samples)
sst_hr2_sum = np.zeros(N_samples)
n_samples_sst_hr = np.zeros(N_samples)

sst_lr_sum = np.zeros(N_samples)
sst_lr2_sum = np.zeros(N_samples)
n_samples_sst_lr = np.zeros(N_samples)


for sample in range(N_samples):
    if sample%100==0:
        print(sample)
    # trying = True
    # while trying:
    r = np.random.randint(0,5585)
    raw_dir = f'raw/{r}/'
    date_test = random.choice(train_dates)
    raw_files = os.listdir(raw_dir)
    ssh_files = [f for f in raw_files if 'ssh_tracks' in f]
    sst_hr_files = [f for f in raw_files if 'sst_hr' in f]
    sst_lr_files = [f for f in raw_files if 'sst_lr' in f]
    
    # ssh_files = [f for f in ssh_files if f'{date_test}' in f]
    # sst_hr_files = [f for f in sst_hr_files if f'{date_test}' in f]
    # sst_lr_files = [f for f in sst_lr_files if f'{date_test}' in f]
    # if len(ssh_files)>0:
    try:
        ssh_out = np.load(raw_dir+random.choice(ssh_files))
        sla = ssh_out[:,-1]
        if np.size(sla[sla!=0])>10:
            # trying = False
            x = ssh_out[:,0].copy()
            x[x!=0] = ((x[x!=0]+0.5*960e3)/960e3)*(128-1)
            y = ssh_out[:,1].copy()
            y[y!=0] = ((-y[y!=0]+0.5*960e3)/960e3)*(128-1)
            sla = ssh_out[:,-1].copy()
            sla = sla[sla!=0].flatten()
            sla = sla[~np.isnan(sla)]
            n_samples[sample] = sla.shape[0]
            sla_sum[sample] = np.sum(sla)
            sla2_sum[sample] = np.sum(sla**2)
        sst_lr = np.load(raw_dir+random.choice(sst_lr_files))
        sst_lr = sst_lr[sst_lr!=0].flatten()
        sst_lr = sst_lr[sst_lr>273]
        sst_lr = sst_lr[~np.isnan(sst_lr)]
        sst_lr_sum[sample] = np.sum(sst_lr)
        sst_lr2_sum[sample] = np.sum(sst_lr**2)
        n_samples_sst_lr[sample] = sst_lr.shape[0]
        
        
        sst_hr = np.load(raw_dir+random.choice(sst_hr_files))
        sst_hr = sst_hr[sst_hr!=0].flatten()
        sst_hr = sst_hr[sst_hr>273]
        sst_hr = sst_hr[~np.isnan(sst_hr)]
        sst_hr_sum[sample] = np.sum(sst_hr)
        sst_hr2_sum[sample] = np.sum(sst_hr**2)
        n_samples_sst_hr[sample] = sst_hr.shape[0]
        
    except:
        print('failed')
        pass
    
    
    
#     dx = (np.roll(x, shift = 1, axis = -1) -np.roll(x, shift = -1, axis = -1))/2
#     dy = (np.roll(y, shift = 1, axis = -1) - np.roll(y, shift = -1, axis = -1))/2
#     dl = (dx**2+dy**2)**0.5

#     # dl = np.stack([dl],axis=-1)
#     dy_pred = (np.roll(sla, shift = 1, axis = -1) - np.roll(sla, shift = -1, axis = -1))/(2*dl+1e-10)
#     # dy_true = (tf.roll(y_true_loss, shift = 1, axis = -2) - tf.roll(y_true_loss, shift = -1, axis = -2))/(2*dl+keras.backend.epsilon())
#     dy_pred = dy_pred[y!=0]
#     # dy_true =dy_true*tf.cast((y_true_loss!=0), dtype='float32')

#     d2y_pred = (np.roll(sla, shift = 1, axis = -1) - 2*sla + np.roll(sla, shift = -1, axis = -1))/(dl**2+1e-10)
#     # d2y_true = (tf.roll(y_true_loss, shift = 1, axis = -2) - 2*y_true_loss + tf.roll(y_true_loss, shift = -1, axis = -2))/(dl**2+keras.backend.epsilon())
#     d2y_pred = d2y_pred[y!=0]
    
    # dsla_std[sample] = np.nanstd(dy_pred)
    # d2sla_std[sample] = np.nanstd(d2y_pred)
    
    # d2y_true =d2y_true*tf.cast((y_true_loss!=0), dtype='float32')


sla_sum = sla_sum[n_samples!=0]
sla2_sum = sla2_sum[n_samples!=0]
n_samples = n_samples[n_samples!=0]

sst_lr_sum = sst_lr_sum[n_samples_sst_lr!=0]
sst_lr2_sum = sst_lr2_sum[n_samples_sst_lr!=0]
n_samples_sst_lr = n_samples_sst_lr[n_samples_sst_lr!=0]

sst_hr_sum = sst_hr_sum[n_samples_sst_hr!=0]
sst_hr2_sum = sst_hr2_sum[n_samples_sst_hr!=0]
n_samples_sst_hr = n_samples_sst_hr[n_samples_sst_hr!=0]

np.save('sla_sum.npy', sla_sum)
np.save('sla2_sum.npy', sla2_sum)
np.save('ssh_stats_n_samples.npy', n_samples)

np.save('sst_lr_sum.npy', sst_lr_sum)
np.save('sst_lr2_sum.npy', sst_lr2_sum)
np.save('sst_lr_stats_n_samples.npy', n_samples_sst_lr)

np.save('sst_hr_sum.npy', sst_hr_sum)
np.save('sst_hr2_sum.npy', sst_hr2_sum)
np.save('sst_hr_stats_n_samples.npy', n_samples_sst_hr)

glob_sla_mean = np.sum(sla_sum)/np.sum(n_samples)
glob_sla_std = np.sqrt(np.sum(sla2_sum)/np.sum(n_samples)-glob_sla_mean**2)

glob_sst_lr_mean = np.sum(sst_lr_sum)/np.sum(n_samples_sst_lr)
glob_sst_lr_std = np.sqrt(np.sum(sst_lr2_sum)/np.sum(n_samples_sst_lr)-glob_sst_lr_mean**2)

glob_sst_hr_mean = np.sum(sst_hr_sum)/np.sum(n_samples_sst_hr)
glob_sst_hr_std = np.sqrt(np.sum(sst_hr2_sum)/np.sum(n_samples_sst_hr)-glob_sst_hr_mean**2)


print('SSH mean:')
print(glob_sla_mean)

print('SSH std:')
print(glob_sla_std)

print('SST LR mean:')
print(glob_sst_lr_mean)

print('SST LR std:')
print(glob_sst_lr_std)

print('SST HR mean:')
print(glob_sst_hr_mean)

print('SST HR std:')
print(glob_sst_hr_std)


# np.save('dsla_std.npy', dsla_std)
# np.save('d2sla_std.npy', d2sla_std)


# print(np.nanmean(sla_mean))
# print(np.nanmean(sla_std))
# print(np.nanmean(dsla_std))
# print(np.nanmean(d2sla_std))
