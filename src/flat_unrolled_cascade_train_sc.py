#!/usr/bin/env python
# coding: utf-8

# # Flat Unrolled Cascade - Single-channel - Train
#
# - Single-channel data
# - Images are 256x256
# - R=5

# In[109]:


import sys
get_ipython().system('{sys.executable} -m pip install nibabel')
get_ipython().system('{sys.executable} -m pip install scipy')
get_ipython().system('{sys.executable} -m pip install scikit-learn')
get_ipython().system('{sys.executable} -m pip install pandas')

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
import tensorflow as tf
# Importing callbacks and data augmentation utils
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import  Adam


# In[25]:


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# In[141]:


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
file_dic = {patient_ids[i]:file_ids[i] for i in range(len(file_ids))}
serie = pd.Series(file_ids, index=patient_ids)


serie_train = serie.loc[[6,9,13,27,33,37,40,41,57,61]]
serie_val = serie.loc[[15,26,39,47,48,56]]
#serie_test = serie.loc[[4,21,24,29,31]]

rec_files_train = serie_train.to_numpy(dtype=str)
rec_files_val = serie_val.to_numpy(dtype=str)

# assign training and validation files
indexes = np.arange(rec_files_train.size,dtype = int)
np.random.shuffle(indexes)
rec_files_train = rec_files_train[indexes]

# Get number of training samples
ntrain = 0
for ii in range(len(rec_files_train)):
    rec = nib.load(file_path + rec_files_train[ii])
    if rec.shape[0] == W:
        aux = rec.shape[-1]
    else:
        aux = rec.shape[0]
    #print(aux, rec.shape)
    ntrain += aux
print('number of training sample', ntrain)

# Load train data
rec_train = np.zeros((ntrain,H,W,2))
kspace_train = np.zeros((ntrain,H,W,2))
aux_counter = 0
for ii in range(len(rec_files_train)):
    aux_rec = nib.load(file_path + rec_files_train[ii]).get_fdata()
    aux_rec = aux_rec / np.abs(aux_rec).max()
    aux_shape = aux_rec.shape
    if aux_shape[0] == W:
        aux_rec = np.swapaxes(aux_rec, 0,2)
        aux = aux_rec.shape[0]
    else:
        aux = aux_rec.shape[0]
        aux_rec = np.transpose(aux_rec,(0,2,1))

    #convert reconstructed data to kspace
    f = np.fft.fft2(aux_rec)

    #convert kspace back to image domain to get complex reconsturction
    aux2_rec = np.zeros((aux_rec.shape[0],aux_rec.shape[1],aux_rec.shape[2],2))
    rec = np.fft.ifft2(f)
    #compile all training data
    rec_train[aux_counter:aux_counter+aux,:,:,0] = rec.real
    rec_train[aux_counter:aux_counter+aux,:,:,1] = rec.imag
    kspace_train[aux_counter:aux_counter+aux,:,:,0] = f.real
    kspace_train[aux_counter:aux_counter+aux,:,:,1] = f.imag
    aux_counter += aux

# Get number of validation samples
nval = 0
for ii in range(len(rec_files_val)):
    rec = nib.load(file_path + rec_files_val[ii])
    if rec.shape[0] == W:
        aux = rec.shape[-1]
    else:
        aux = rec.shape[0]
    #print(aux, rec.shape)
    nval += aux
print('number of validation samples', nval)

# Load valiadtion data
kspace_val = np.zeros((nval,H,W,2))
rec_val = np.zeros((nval,H,W,2))

aux_counter = 0
for ii in range(len(rec_files_val)):
    aux_rec = nib.load(file_path + rec_files_val[ii]).get_fdata()
    aux_rec = aux_rec / aux_rec.max()
    aux_shape = aux_rec.shape
    if aux_shape[0] == W:
        aux_rec = np.swapaxes(aux_rec, 0,2)
        aux = aux_rec.shape[0]
    else:
        aux = aux_rec.shape[0]
        aux_rec = np.transpose(aux_rec,(0,2,1))


    #convert reconstructed data to kspace
    f = np.fft.fft2(aux_rec)
    #fshift = np.fft.fftshift(f)
    aux_kspace = np.zeros((aux_rec.shape[0],aux_rec.shape[1],aux_rec.shape[2],2))
    aux_kspace[:,:,:,0] = f.real
    aux_kspace[:,:,:,1] = f.imag
    #convert kspace back to image domain to get complex reconsturction
    aux2_rec = np.zeros((aux_rec.shape[0],aux_rec.shape[1],aux_rec.shape[2],2))
    rec = np.fft.ifft2(f)
    #compile all training data
    rec_val[aux_counter:aux_counter+aux,:,:,0] = rec.real
    rec_val[aux_counter:aux_counter+aux,:,:,1] = rec.imag
    kspace_val[aux_counter:aux_counter+aux,:,:,0] = f.real
    kspace_val[aux_counter:aux_counter+aux,:,:,1] = f.imag

    aux_counter += aux

# Loading sampling patterns. Notice that here we are using uncentred k-space
var_sampling_mask = np.fft.fftshift(~np.load("../data/R5_256x256_poisson_center_true_radius_40.npy"),axes=(1,2))
var_sampling_mask = np.concatenate((var_sampling_mask[:,:,:,np.newaxis],var_sampling_mask[:,:,:,np.newaxis]),
                                   axis = -1)


epochs = 20
batch_size= 5
os.makedirs('../models/', exist_ok=True)
model_name = "../models/flat_unrolled_cascade_ikikii.hdf5"

# Early stopping callback to shut down training after
# 5 epochs with no improvement
earlyStopping = EarlyStopping(monitor='val_loss',
                                       patience=20,
                                       verbose=0, mode='min')

# Checkpoint callback to save model  along the epochs
checkpoint = ModelCheckpoint(model_name, mode = 'min',
                            monitor='val_loss',
                            verbose=0,
                            save_best_only=True,
                            save_weights_only = True)


# On the fly data augmentation
def combine_generator(gen1,gen2,under_masks):
    while True:
        rec_real = gen1.next()
        rec_imag = gen2.next()
        f = np.fft.fft2(rec_real[:,:,:,0]+1j*rec_imag[:,:,:,0])
        #kspace = np.fft.ifftshift(f)
        kspace2 = np.zeros((f.shape[0],f.shape[1],f.shape[2],2))
        kspace2[:,:,:,0] = f.real
        kspace2[:,:,:,1] = f.imag
        indexes = np.random.choice(np.arange(under_masks.shape[0], dtype=int), rec_real.shape[0], replace=False)
        kspace2[under_masks[indexes]] = 0

        rec_complex = np.zeros((rec_real.shape[0],rec_real.shape[1],rec_real.shape[2],2),dtype = np.float32)
        rec_complex[:,:,:,0] = rec_real[:,:,:,0]
        rec_complex[:,:,:,1] = rec_imag[:,:,:,0]

        yield([kspace2,under_masks[indexes].astype(np.float32)],[rec_complex])

seed = 905
image_datagen1 = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

image_datagen2 = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')


image_generator1 = image_datagen1.flow(rec_train[:,:,:,0,np.newaxis],batch_size = batch_size,seed = seed, shuffle=False)
image_generator2 = image_datagen2.flow(rec_train[:,:,:,1,np.newaxis],batch_size = batch_size,seed = seed, shuffle=False)


combined = combine_generator(image_generator1,image_generator2, var_sampling_mask)

# Undersampling the validation set
indexes = np.random.choice(np.arange(var_sampling_mask.shape[0],dtype =int),kspace_val.shape[0],replace = True)
val_var_sampling_mask = (var_sampling_mask[indexes])
kspace_val[val_var_sampling_mask] = 0


# Training our model
model = fsnet.deep_cascade_flat_unrolled("ikikii", H, W,channels=2)
opt = Adam(learning_rate = 1e-3,decay = 1e-4)
model.compile(loss = 'mse',optimizer=opt)
print(model.summary())

hist = model.fit(combined,
             epochs=epochs,
             steps_per_epoch=rec_train.shape[0]//batch_size,
             verbose=1,
             validation_data= ([kspace_val,val_var_sampling_mask],[rec_val]),
             callbacks=[checkpoint,earlyStopping])
