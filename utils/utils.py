# Matthieu Dagommer
# 27/06/2022

import sys
sys.path.append("C:/Users/matth/Documents/Martinos Center/mrtoct") 
import os
import pickle
import argparse
from sqlite3 import Row
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis, skew
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.layers import Layer
from keras.layers import InputSpec
from numba import njit
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import importlib
#from models.base_model import BaseModel

def RetrieveDataSetOptions(parser, dataset_name):

    if os.name == 'nt':
        path = "C:/Users/matth/Documents/Martinos Center/mrtoct/datasets/"
    else:
        path = "/autofs/space/guerin/USneuromod/MATHIEU/mrtoct/datasets/"
    dataset = pickle.load(open(path + dataset_name + "/dataset_info", "rb"))   

    parser.set_defaults(seed = dataset["seed"], no_sqrt = dataset["no_sqrt"], \
        no_mask = dataset["no_mask"], mm_ct = dataset["mm_ct"], \
        mm_t1 = dataset["mm_t1"], set_pct = dataset["set_pct"], #patch25d = dataset["patch25d"],
        ns_valid = dataset["ns_valid"], patch_shape = dataset["patch_shape"], tanh = dataset["tanh"])
    
    return parser

@njit
def generate_coords(n_subjects, batch_size, to, no_flip, patch_shape):
    three_d = False
    if len(patch_shape) == 4: # 3D patch
        n_chans = patch_shape[-1]
        patch_shape = patch_shape[:-1]
        three_d = True
    # generate random coordinates to retrieve batch from training set
    subjects = np.random.randint(0, n_subjects, batch_size)
    if to:
        planes = np.zeros((batch_size), dtype = np.int64) + 2
    else:
        planes = np.random.randint(0, 3, batch_size) # 0: sagittal; 1: coronal; 2: transverse 
    min_plane = np.array([60, 40, 90]) # minimum slice for sagittal, coronal and transverse planes
    max_plane = np.array([200, 230, 230])
    

    slices = np.zeros((3, batch_size), dtype = np.int64)
    w_down = np.zeros((3,batch_size), dtype = np.int64)
    w_up = np.zeros((3,batch_size), dtype = np.int64)
    #for i in range(batch_size):
    #    slices[i] = np.random.randint(min_plane[planes[i]],max_plane[planes[i]])

    for j in range(batch_size):
        for i in range(3):
    #for i in range(batch_size):
        #slices[i] = np.random.randint(min_plane[planes[i]], max_plane[planes[i]])
            slices[i,j] = np.random.randint(min_plane[i], max_plane[i])
            

        # Set the lower boundary and the upper boundary

        # Boundaries for secondary planes
        if planes[j] == 0:
            w_up[1,j] = w_up[2,j] = patch_shape[0] / 2
        elif planes[j] == 1:
            w_up[0,j] = w_up[2,j] = patch_shape[0] / 2
        elif planes[j] == 2:
            w_up[0,j] = w_up[1,j] = patch_shape[0] / 2
        
        w_down[:,j] = w_up[:,j] - 1

        # Boundaries for main plane (which is planes[j])
        if patch_shape[-1] != 1:
            w_up[planes[j],j] = patch_shape[-1] / 2
            w_down[planes[j],j] = w_up[planes[j],j] - 1
        else:
            w_up[planes[j],j] = 0
            w_down[planes[j],j] = 0
        
        
            

        # Modify slices if the cropped patch goes beyond the boundaries
        for i in range(3):
            if slices[i,j] + w_up[i,j] + 1 > 256 or slices[i,j] - w_down[i,j] < 0:
                if i == planes[j] and patch_shape[-1] == 256:
                    #if patch_shape[-1] == 256:
                    w_down[i,j] = slices[i,j]
                    w_up[i,j] = 255 - slices[i,j]
                elif 256 in patch_shape:
                    #if patch_shape[0] == 256:
                    w_down[i,j] = slices[i,j]
                    w_up[i,j] = 255 - slices[i,j]
                else:
                    if slices[i,j] + w_up[i,j] + 1 > 256 : 
                        slices[i,j] = 255 - w_up[i,j]
                    elif slices[i,j] - w_down[i,j] < 0 :
                        slices[i,j] = w_down[i,j]
          


    orientations = np.random.randint(0, 4, batch_size)

    if not no_flip:
        flips = np.random.randint(0, 2, batch_size)
    else:
        flips = np.zeros((batch_size), dtype = np.int64)
    
    batch_coordinates = np.concatenate((subjects.reshape(1,-1), planes.reshape(1,-1), slices.reshape(3,-1), \
                                        orientations.reshape(1,-1), flips.reshape(1,-1)), axis = 0)
    return batch_coordinates

def load_samples(x, y, m, n_subjects, batch_size, disc_shape, to, no_flip, patch_shape, pct = False, coords = None):
    if coords is None:
        coords = generate_coords(n_subjects, batch_size, to, no_flip, patch_shape)
    x_batch = retrieve_slices(coords, batch_size, x, patch_shape, pct = pct)
    y_batch = retrieve_slices(coords, batch_size, y, patch_shape, pct = False)
    m_batch = retrieve_slices(coords, batch_size, m, patch_shape, pct = False)
    y_ = np.ones((batch_size, disc_shape, disc_shape, 1))
    return x_batch, y_batch, m_batch, coords, y_

#@njit
def generate_real_samples(X_train, Y_train, batch_size, disc_shape, N_subjects, transverse_only, no_flip, set_pct, patch_shape):
    
    three_d = False
    if len(patch_shape) == 4: # 3D patch
        n_chans = patch_shape[-1]
        patch_shape = patch_shape[:-1]
        three_d = True

    
    subjects = np.random.randint(0, N_subjects, batch_size)
    
    if transverse_only:
        planes = np.zeros((batch_size), dtype = np.int64) + 2
    else:
        planes = np.random.randint(0, 3, batch_size) # 0: sagittal; 1: coronal; 2: transverse 
        
    min_plane = np.array([60, 40, 90]) # minimum slice for sagittal, coronal and transverse planes
    max_plane = np.array([200, 230, 230])

    slices = np.zeros((3,batch_size), dtype = np.int64)
    w_down = np.zeros((3,batch_size), dtype = np.int64)
    w_up = np.zeros((3,batch_size), dtype = np.int64)

    for j in range(batch_size):
        for i in range(3):
            slices[i,j] = np.random.randint(min_plane[i], max_plane[i])
            

        # Set the lower boundary and the upper boundary

        # Boundaries for secondary planes
        if planes[j] == 0:
            w_up[1,j] = w_up[2,j] = patch_shape[0] / 2
        elif planes[j] == 1:
            w_up[0,j] = w_up[2,j] = patch_shape[0] / 2
        elif planes[j] == 2:
            w_up[0,j] = w_up[1,j] = patch_shape[0] / 2
        
        w_down[:,j] = w_up[:,j] - 1

        # Boundaries for main plane (which is planes[j])
        if patch_shape[-1] != 1:
            w_up[planes[j],j] = patch_shape[-1] / 2
            w_down[planes[j],j] = w_up[planes[j],j] - 1
        else:
            w_up[planes[j],j] = 0
            w_down[planes[j],j] = 0
        
        
        # Modify slices if the cropped patch goes beyond the boundaries
        for i in range(3):
            if slices[i,j] + w_up[i,j] + 1 > 256 or slices[i,j] - w_down[i,j] < 0:
                if i == planes[j] and patch_shape[-1] == 256:
                    w_down[i,j] = slices[i,j]
                    w_up[i,j] = 255 - slices[i,j]
                elif 256 in patch_shape:
                    w_down[i,j] = slices[i,j]
                    w_up[i,j] = 255 - slices[i,j]
                else:
                    if slices[i,j] + w_up[i,j] + 1 > 256 : 
                        slices[i,j] = 255 - w_up[i,j]
                    elif slices[i,j] - w_down[i,j] < 0 :
                        slices[i,j] = w_down[i,j]
                
                
    orientations = np.random.randint(0, 4, batch_size)
    
    if not no_flip:
        flips = np.random.randint(0, 2, batch_size)
    else:
        flips = np.zeros((batch_size), dtype = np.int64)
    
    batch_coordinates = np.concatenate((subjects.reshape(1,-1), planes.reshape(1,-1), slices.reshape(3,-1), \
                                        orientations.reshape(1,-1), flips.reshape(1,-1)), axis = 0)


    for i in range(batch_size):
        
        X_batch = X_train[
            subjects[i],
            slices[0,i]-w_down[0,i]:slices[0,i]+w_up[0,i]+1,
            slices[1,i]-w_down[1,i]:slices[1,i]+w_up[1,i]+1,
            slices[2,i]-w_down[2,i]:slices[2,i]+w_up[2,i]+1,
            :
            ]

        Y_batch = Y_train[
            subjects[i],
            slices[0,i]-w_down[0,i]:slices[0,i]+w_up[0,i]+1,
            slices[1,i]-w_down[1,i]:slices[1,i]+w_up[1,i]+1,
            slices[2,i]-w_down[2,i]:slices[2,i]+w_up[2,i]+1,
            :
            ]
        
        if planes[i] == 0: # sagittal
            X_batch = np.transpose(X_batch, (1,2,0,3))
            Y_batch = np.transpose(Y_batch, (1,2,0,3))
        elif planes[i] == 1: # coronal
            X_batch = np.transpose(X_batch, (0,2,1,3))
            Y_batch = np.transpose(Y_batch, (0,2,1,3))

        X_batch = np.rot90(X_batch, k=orientations[i])
        Y_batch = np.rot90(Y_batch, k=orientations[i])

        if flips[i] == 1:
            X_batch = np.flip(X_batch)
            Y_batch = np.flip(Y_batch)
        
        if not three_d:
            if i == 0:
                if set_pct:
                    X_batch_concat = X_batch.copy().reshape(1,patch_shape[0],patch_shape[1],patch_shape[2]*2)
                else:
                    X_batch_concat = X_batch.copy().reshape(1,patch_shape[0],patch_shape[1],patch_shape[2])
                Y_batch_concat = Y_batch.copy().reshape(1,patch_shape[0],patch_shape[1],patch_shape[2])
            else:
                if set_pct:
                    X_batch_concat = np.concatenate((X_batch_concat, X_batch.copy().reshape(1,patch_shape[0],patch_shape[1],patch_shape[2]*2)), axis = 0)
                else:
                    X_batch_concat = np.concatenate((X_batch_concat, X_batch.copy().reshape(1,patch_shape[0],patch_shape[1],patch_shape[2])), axis = 0)
                Y_batch_concat = np.concatenate((Y_batch_concat, Y_batch.copy().reshape(1,patch_shape[0],patch_shape[1],patch_shape[2])), axis = 0)
        else:
            if i == 0:
                if set_pct:
                    X_batch_concat = X_batch.copy().reshape(1,patch_shape[0],patch_shape[1],patch_shape[2],n_chans*2)
                else:
                    X_batch_concat = X_batch.copy().reshape(1,patch_shape[0],patch_shape[1],patch_shape[2],n_chans)
                Y_batch_concat = Y_batch.copy().reshape(1,patch_shape[0],patch_shape[1],patch_shape[2],n_chans)
            else:
                if set_pct:
                    X_batch_concat = np.concatenate((X_batch_concat, X_batch.copy().reshape(1,patch_shape[0],patch_shape[1],patch_shape[2],n_chans*2)), axis = 0)
                else:
                    X_batch_concat = np.concatenate((X_batch_concat, X_batch.copy().reshape(1,patch_shape[0],patch_shape[1],patch_shape[2],n_chans)), axis = 0)
                Y_batch_concat = np.concatenate((Y_batch_concat, Y_batch.copy().reshape(1,patch_shape[0],patch_shape[1],patch_shape[2],n_chans)), axis = 0)
        disc_y_shape = tuple([batch_size] + list(disc_shape))
        disc_y = np.ones(shape=disc_y_shape)
    return X_batch_concat, Y_batch_concat, batch_coordinates, disc_y


#@njit
def retrieve_slices(batch_coordinates, batch_size, X, patch_shape, pct = False):

    three_d = False
    if len(patch_shape) == 4: # 3D patch
        n_chans = patch_shape[-1]
        patch_shape = patch_shape[:-1]
        three_d = True

    subjects = batch_coordinates[0]
    planes = batch_coordinates[1]
    slices = batch_coordinates[2:5]
    orientations = batch_coordinates[5]
    flips = batch_coordinates[6]

    w_up = np.zeros((3, batch_size), dtype = np.int64)
    w_down = np.zeros((3, batch_size), dtype = np.int64)

    for j in range(batch_size):

        # Boundaries for secondary planes
        if planes[j] == 0:
            w_up[1,j] = w_up[2,j] = patch_shape[0] / 2
        elif planes[j] == 1:
            w_up[0,j] = w_up[2,j] = patch_shape[0] / 2
        elif planes[j] == 2:
            w_up[0,j] = w_up[1,j] = patch_shape[0] / 2
        
        w_down[:,j] = w_up[:,j] - 1

        # Boundaries for main plane (which is planes[j])
        if patch_shape[-1] != 1:
            w_up[planes[j],j] = patch_shape[-1] / 2
            w_down[planes[j],j] = w_up[planes[j],j] - 1
        else:
            w_up[planes[j],j] = 0
            w_down[planes[j],j] = 0
        
        for i in range(3):
            if slices[i,j] + w_up[i,j] + 1 > 256 or slices[i,j] - w_down[i,j] < 0:
                if i == planes[j] and patch_shape[-1] == 256:
                    w_down[i,j] = slices[i,j]
                    w_up[i,j] = 255 - slices[i,j]
                elif 256 in patch_shape:
                    w_down[i,j] = slices[i,j]
                    w_up[i,j] = 255 - slices[i,j]

    for i in range(batch_size):
        X_batch = X[
            subjects[i],
            slices[0,i]-w_down[0,i]:slices[0,i]+w_up[0,i]+1,
            slices[1,i]-w_down[1,i]:slices[1,i]+w_up[1,i]+1,
            slices[2,i]-w_down[2,i]:slices[2,i]+w_up[2,i]+1,
            :
            ]

        if planes[i] == 0:
            X_batch = np.transpose(X_batch, (1,2,0,3))
        elif planes[i] == 1:
            X_batch = np.transpose(X_batch, (0,2,1,3))
        
        X_batch = np.rot90(X_batch, k = orientations[i])

        if flips[i] == 1:
            X_batch = np.flip(X_batch)

        if not three_d:
            if i == 0:
                if pct:
                    X_concat = X_batch.copy().reshape(1,patch_shape[0],patch_shape[1],patch_shape[2]*2)
                else:
                    X_concat = X_batch.copy().reshape(1,patch_shape[0],patch_shape[1],patch_shape[2])
            else:
                if pct:
                    #X_concat = X_batch.copy().reshape(1,patch_shape[0],patch_shape[1],patch_shape[2]*2)
                    X_concat = np.concatenate((X_concat, X_batch.copy().reshape(1,patch_shape[0],patch_shape[1],patch_shape[2]*2)), axis = 0)
                else:
                    X_concat = np.concatenate((X_concat, X_batch.copy().reshape(1,patch_shape[0],patch_shape[1],patch_shape[2])), axis = 0)
        else:
            if i == 0:
                if pct:
                    X_concat = X_batch.copy().reshape(1,patch_shape[0],patch_shape[1],patch_shape[2],n_chans*2)
                else:
                    X_concat = X_batch.copy().reshape(1,patch_shape[0],patch_shape[1],patch_shape[2],n_chans)
            else:
                if pct:
                    #X_concat = X_batch.copy().reshape(1,patch_shape[0],patch_shape[1],patch_shape[2]*2)
                    X_concat = np.concatenate((X_concat, X_batch.copy().reshape(1,patch_shape[0],patch_shape[1],patch_shape[2],n_chans*2)), axis = 0)
                else:
                    X_concat = np.concatenate((X_concat, X_batch.copy().reshape(1,patch_shape[0],patch_shape[1],patch_shape[2],n_chans)), axis = 0)
    return X_concat



def save_info(opt, elapsed_time, mae, mm_ct):

    model_info = vars(opt)
    # poro_range is the range of output used for training
    model_info['poro_range'] = [mm_ct.data_min_[0], mm_ct.data_max_[0]]
    model_info['elapsed_time'] = elapsed_time
    model_info['best_epoch_mae'] = mae

    pickle.dump(model_info, open(opt.model_path + opt.name +"_info", "wb"))

    model_info_txt = ""
    for i, (key, item) in enumerate(model_info.items()):
        if i > 0:
            model_info_txt = model_info_txt + ", "
        model_info_txt = model_info_txt + str(key) + ": " + str(item)
    
    with open(opt.model_path + opt.name + "_info.txt", "w") as f:
        f.write(model_info_txt)



def learning_curve(opt, temp_output):
    fig = plt.figure(figsize = (22,6))
    fig.subplots_adjust(hspace=0.4, wspace = 0.4)
    ax = fig.add_subplot(1,2,1)

    # Generate x-axis
    x = np.linspace(0, (opt.save_freq-1)*opt.save_every_x_batches / opt.bat_per_epo, opt.save_freq)
    n_temp_epochs = (len(temp_output[0])-1) // opt.save_freq
    temp_x = []
    for i in range(n_temp_epochs):
        temp_x = temp_x + (x+i).tolist()
    temp_x.append(n_temp_epochs)
    ax.plot(temp_x, temp_output[0], 'r', label = "Training", linewidth = 3.5)
    ax.plot(temp_x, temp_output[1], 'b', label = "Validation", linewidth = 3.5)
    ax.set_xlabel("Epochs", fontsize = 12)
    ax.set_ylabel("MSE", fontsize = 12)
    plt.legend(fontsize = 12)

    plt.gcf()
    plt.savefig(opt.model_path + "learning_curve.jpg")



def plot_images(model, model_path, test_set, mm_ct):

    src_img, gen_img, tar_img, src_img_mask, \
        gen_img_mask, tar_img_mask, vmin, vmax, ix = generate_images(model, test_set, mm_ct)

    images = np.vstack((src_img, gen_img, tar_img, src_img_mask, gen_img_mask, tar_img_mask))
    titles = ['Source MRI', 'Generated Poro', 'Expected Poro', 'Source MRI + Mask', 'Generated Poro + Mask', 'Expected Poro + Mask']
    for i in range(len(images)):
        plt.subplot(2, 3, 1 + i)
        plt.axis('off')
        plt.imshow(images[i], vmin = vmin[i], vmax = vmax[i])
        if i == 0:
            plt.annotate("Slice n° {}".format(ix), xy = (10,20), fontsize = 20, color ='white')
        plt.title(titles[i], fontsize = 15)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.gcf()
    plt.savefig(model_path + "output.png")


def generate_images(model, test_set, mm_ct):

    X, Y, M = test_set

    # select random example
    ix = np.random.randint(0, len(X), 1)
    src_image, tar_image, mask_image = X[ix], Y[ix], M[ix]
    src_image_mask = tf.math.multiply(src_image, mask_image)

    # generate image from source
    gen_image = model.predict(src_image)
    src_image = np.array(src_image); tar_image = np.array(tar_image); mask_image = np.array(mask_image); gen_image = np.array(gen_image)

    gen_image_ = mm_ct.inverse_transform(gen_image.flatten().reshape(-1,1))
    gen_image = np.reshape(gen_image_, gen_image.shape)
    gen_image_mask = tf.math.multiply(gen_image - 1, mask_image) + 1

    tar_image_ = mm_ct.inverse_transform(tar_image.flatten().reshape(-1,1))
    tar_image = np.reshape(tar_image_, tar_image.shape)
    tar_image_mask = tf.math.multiply(tar_image - 1, mask_image) + 1

    # Scale mins and maxs
    vmin = np.zeros((6)); vmin[:] = mm_ct.data_min_[0]; #vmin[0] = mm_t1.data_min_[0]; vmin[3] = mm_t1.data_min_[0]
    vmax = np.zeros((6)); vmax[:] = mm_ct.data_max_[0]; #vmax[0] = mm_t1.data_max_[0]; vmax[3] = mm_t1.data_max_[0]

    return src_image, gen_image, tar_image, src_image_mask, gen_image_mask, tar_image_mask, vmin, vmax, ix



def distrib(opt, temp_output):

    bevel = temp_output[2]
    N_valid = len(bevel)

    # compute statistical features
    mean = float(np.mean(bevel))
    std_dev = float(np.std(bevel))
    kurt = float(kurtosis(bevel))
    skewn = float(skew(bevel))
    q1 = float(np.quantile(bevel, 0.25))
    median = float(np.quantile(bevel, 0.5))
    q3 = float(np.quantile(bevel, 0.75))
    mae = np.absolute(bevel).sum() / N_valid
    mse = np.square(bevel).sum() / N_valid

    text = "mean: {:.3f}\nstd: {:.3f}\nkurt: {:.3f}\nskew: {:.3f}\nq1: {:.3f}\nmed: {:.3f}\nq3: {:.3f}\niqr: {:.3f}\nmae: {:.3f}\nmse: {:.3f}\n".format(mean, std_dev, kurt, skewn, q1, median, q3, q3-q1, mae, mse)

    fig, ax = plt.subplots(figsize = (10, 8))
    hist_out = ax.hist(bevel, bins = 100, range = (-1,1))

    max_hist = float(max(hist_out[0]))

    ax.text(-1, max_hist/2, text,
            color='green', fontsize=15)

    ax.set(title = "Distribution of voxel porosity validation error " + opt.name, xlabel = "values", ylabel = "occurences")
    plt.gcf()
    plt.savefig(opt.model_path + "distrib.jpg")



@njit
def prepare_data(ct_, pct_, t1_, mask_, threshold, no_mask, no_sqrt, mask_opt):
    
    print(" =====> Setting the masks..")
    # Replacing all non-zero values of the mask with 1 
    if not no_mask:
        mask_ = np.where(mask_ > threshold, 1, 0)
    else:
        mask_ = np.where(mask_ > 0, 1, 0)

    print(" =====> CT: Converting HU to porosity..")
    # CT: Converting Hounsfield units to bone porosity
    ct_ = 1 - ct_/1000
    ct_ = np.where(ct_ > 1, 1, ct_)
    ct_ = np.where(ct_ < 0, 0, ct_)
    
    print(" ====> pCT: Converting HU to porosity..")
    # pCT: Converting Hounsfield units to bone porosity
    pct_ = 1 - pct_/1000
    pct_ = np.where(pct_ > 1, 1, pct_)
    pct_ = np.where(pct_ < 0, 0, pct_)

    print(" =====> Few last steps..")
    if not no_sqrt:
        ct_ = np.sqrt(ct_)
        pct_ = np.sqrt(pct_)

    # Applying mask on input data
    if not no_mask:
        if mask_opt == 0: # All mask
            ct_ = np.reshape(np.multiply(ct_ - 1, mask_) + 1, ct_.shape)
            pct_ = np.reshape(np.multiply(pct_ - 1, mask_) + 1, pct_.shape)
            t1_ = np.reshape(np.multiply(t1_, mask_), t1_.shape)
        elif mask_opt == 1: # mask CT only
            ct_ = np.reshape(np.multiply(ct_ - 1, mask_) + 1, ct_.shape)
            pct_ = np.reshape(np.multiply(pct_ - 1, mask_) + 1, pct_.shape)
        elif mask_opt == 2: # mask MRI only
            t1_ = np.reshape(np.multiply(t1_, mask_), t1_.shape)
    
    return t1_, ct_, pct_, mask_


def normalize_dataset(X, Y, set_pct, fit = False, mm_ct = None, mm_t1 = None, tanh = False):
    
    outs = []
    n_subjects = X.shape[0]

    if fit:
        if tanh:
            mm_t1 = MinMaxScaler(feature_range = (-1,1))
            mm_ct = MinMaxScaler(feature_range = (-1,1))
        else:
            mm_t1 = MinMaxScaler()
            mm_ct = MinMaxScaler()

        X_mm = mm_t1.fit_transform(np.expand_dims(X[:,:,:,:,0].flatten(),-1))
        Y_mm = mm_ct.fit_transform(np.expand_dims(Y.flatten(),-1))
    else:
        X_mm = mm_t1.transform(np.expand_dims(X[:,:,:,:,0].flatten(),-1))
        Y_mm = mm_ct.transform(np.expand_dims(Y.flatten(),-1))

    if set_pct:
        X_t1 = np.reshape(X_mm, (n_subjects,256,256,256,1))
        X_pct = mm_ct.fit_transform(np.expand_dims(X[:,:,:,:,1].flatten(),-1))
        X = np.concatenate((X_t1, X_pct.reshape(n_subjects,256,256,256,1)), axis = -1)
    else:
        X = np.reshape(X_mm, X.shape)
    Y = np.reshape(Y_mm, Y.shape)
    
    outs.append(X)
    outs.append(Y)

    if fit:
        outs.append(mm_ct); outs.append(mm_t1)
    
    return outs

def batch_computation(model, input):
        temp_bs = 200 # temporary batch size
        q = input.shape[0] // temp_bs
        r = input.shape[0] % temp_bs
        nb_iters = q if r == 0 else q + 1
        generated = []
        for i in range(nb_iters):
            if i < nb_iters - 1:
                generated.append(model.predict(input[temp_bs*i:temp_bs*(i+1)], verbose = 0))
            else:
                generated.append(model.predict(input[temp_bs*i:], verbose = 0))
        generated = np.concatenate(generated, axis = 0)
        return generated

def tf_gradient_3d(a):
    #*axis = 1
    left = tf.concat([a[:,1:], tf.expand_dims(a[:,-1],1)], axis = 1)
    right = tf.concat([tf.expand_dims(a[:,0],1), a[:,:-1]], axis = 1)
    #tf.cast(left, dtype = tf.float32)
    #tf.cast(right, dtype = tf.float32)

    ones = tf.ones_like(right[:, 2:], tf.float32)
    one = tf.expand_dims(ones[:,0], 1)
    dx = tf.concat((one, ones*2, one), 1)

    gx = (left - right )/dx

    #* axis = 0
    left = tf.concat([a[1:,:], tf.expand_dims(a[-1,:],0)], axis = 0)
    right = tf.concat([tf.expand_dims(a[0,:],0), a[:-1,:]], axis = 0)
   
    ones = tf.ones_like(right[2:], tf.float32)
    one = tf.expand_dims(ones[0], 0)
    dx = tf.concat((one, ones*2, one), 0)

    gy = (left - right )/dx

    # *axis = 2
    left = tf.concat([a[:,:,1:], tf.expand_dims(a[:,:,-1],2)], axis = 2)
    right = tf.concat([tf.expand_dims(a[:,:,0],2), a[:,:,:-1]], axis = 2)
    
    ones = tf.ones_like(right[:, :, 2:], tf.float32)
    one = tf.expand_dims(ones[:,:,0], 2)
    dx = tf.concat((one, ones*2, one), 2)

    gz = (left - right )/dx

    # Added on 01/23
    gx = tf.abs(gx)
    gy = tf.abs(gy)
    gz = tf.abs(gz)

    return gx, gy, gz