#!/usr/bin/env python
# coding: utf-8

# # Flat Unrolled Cascade - Single-channel - Test
# 
# - Single-channel data
# - Images are 256x256
# - R=5 

# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt
import numpy as np
import os
import glob
import sys
import nibabel as nib
import pandas as pd
# Importing our model
MY_UTILS_PATH = "../src/"
if not MY_UTILS_PATH in sys.path:
    sys.path.append(MY_UTILS_PATH)
import cs_models_sc as fsnet

# Importing callbacks and data augmentation utils
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import  Adam


# In[9]:


## PARAMETERS
H,W = 256,256 # Training image dimensions
channels = 2 # complex data 0-> real; 1-> imaginary
norm = np.sqrt(H*W)

#All data
data_path = "../data/reconstructed/*.nii.gz"
all_files = np.asarray(glob.glob(data_path))
print(len(all_files))
file_path = '../data/reconstructed/'

file_ids = np.array([file.split('/')[-1] for file in all_files])

patient_ids = np.array([file[3:5] for file in file_ids], dtype=int)
#np.concatenate([patient_ids[:,np.newaxis], file_ids[:,np.newaxis]],axis=1,dtype=int)
np.unique(patient_ids)

serie = pd.Series(file_ids, index=patient_ids)
serie_test = serie.loc[[4,21,24,29,31]]
rec_files_test = serie_test.to_numpy(dtype=str)


# In[10]:


# Loading sampling patterns. Notice that here we are using uncentred k-space
var_sampling_mask = np.fft.fftshift(~np.load("../data/R5_256x256_poisson_center_true_radius_40.npy")                                     ,axes = (1,2))
var_sampling_mask = np.concatenate((var_sampling_mask[:,:,:,np.newaxis],var_sampling_mask[:,:,:,np.newaxis]),                                          axis = -1)[0]


# White pixels are retrospectively discarded
plt.figure()
plt.imshow(var_sampling_mask[:,:,0],cmap = "gray")
plt.axis("off")
plt.show()

print("Undersampling:", 1.0*var_sampling_mask.sum()/var_sampling_mask.size)


# In[11]:


# Training our model
model_name = "../models/flat_unrolled_cascade_ikikii.hdf5"
model = fsnet.deep_cascade_flat_unrolled("ikikii", H, W)
opt = Adam(learning_rate = 1e-3,decay = 1e-5)
model.compile(loss = 'mse',optimizer=opt)
model.load_weights(model_name)


# In[19]:


si = 100 # slice to display
for ii in range(len(rec_files_test)):
    rec_test = nib.load(file_path + rec_files_test[ii]).get_fdata()/norm
    if rec_test.shape[0] == W:
        aux = rec_test.shape[-1]
        rec_test = np.swapaxes(rec_test, 0,2)
    else:
        aux = rec_test.shape[0]
        rec_test = np.transpose(aux_rec,(0,2,1))
    
    kspace_test = np.zeros((rec_test.shape[0],rec_test.shape[1],rec_test.shape[2],2))
    aux = np.fft.fft2(rec_test)
    kspace_test[:,:,:,0] = aux.real
    kspace_test[:,:,:,1] = aux.imag
    var_sampling_mask_test = np.tile(var_sampling_mask,(kspace_test.shape[0],1,1,1))
    #print(var_sampling_mask_test.shape)
    kspace_test[:,var_sampling_mask] = 0
    pred = model.predict([kspace_test,var_sampling_mask_test])
    
    #name = kspace_files_test[ii].split("/")[-1].split(".npy")[0]
    #np.save(name + "_rec.npy",pred)
    rec_pred = np.abs(pred[si,:,:,0]+1j*pred[si,:,:,1])
    plt.figure(figsize=(16,4))
    plt.subplot(141)
    plt.imshow(rec_pred,cmap = "gray")
    plt.axis("off")
    plt.subplot(142)
    plt.imshow(np.abs(rec_test[si,:,:]),cmap = "gray")
    plt.axis("off")
    plt.subplot(143)
    plt.imshow(np.abs(rec_test[si,:,:] - rec_pred))
    plt.subplot(144)
    plt.imshow(np.log(1+np.abs(kspace_test[si,:,:,0]+1j*kspace_test[si,:,:,1])), cmap='gray')
    plt.show()


# In[ ]:




