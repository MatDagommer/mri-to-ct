# Matthieu Dagommer
# 27/06/2022

import os
import numpy as np
from numpy import load
from numpy import vstack
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
import pickle
from options.preprocessing_options import PreprocessingOptions
from utils import utils


opt = PreprocessingOptions().gather_options()
#print(opt.ns_valid)

# Retrieving data
print("Preparing Data. Please wait...")

if os.name == 'nt':
    path = "C:/Users/matth/Documents/Martinos Center/Subjects/"
else:
    path = "/autofs/space/guerin/USneuromod/MATHIEU/DATA/"
os.chdir(path)

ct_, pct_, t1_, mask_ = [], [], [], []

subjects_list = []
for file in os.listdir():
    if os.path.isdir(file):
        subjects_list.append(file)

print("Reading Data from .nii files...")

for subject in subjects_list:

    print(subject)
    folder_path = path + subject
    file_ct = os.path.join(folder_path, "CTInT1_resliced.nii")
    file_pct = os.path.join(folder_path, "pCTInT1_resliced.nii")
    file_t1 = os.path.join(folder_path, "T1_resliced.nii")
    file_skull = os.path.join(folder_path, "skull_SAMSEG_resliced.nii")

    img_ct = nib.load(file_ct)
    img_pct = nib.load(file_pct)
    img_t1 = nib.load(file_t1)
    sk_mask = nib.load(file_skull)

    img_ct = img_ct.get_fdata()
    img_pct = img_pct.get_fdata()
    img_t1 = img_t1.get_fdata()
    sk_mask = sk_mask.get_fdata()
    
    # Setting shape to (256,256,256,1)
    img_ct = img_ct.reshape(256,256,256,1); img_pct = img_pct.reshape(256,256,256,1) 
    img_t1 = img_t1.reshape(256,256,256,1); sk_mask = sk_mask.reshape(256,256,256,1)
    
    # Storing Data in Lists
    ct_.append(img_ct)
    pct_.append(img_pct)
    t1_.append(img_t1)
    mask_.append(sk_mask)

# Normalising masks 
print("Normalizing Masks...")

mm_mask = MinMaxScaler()
mask_ar = np.array(mask_)
mask_norm = mm_mask.fit_transform(np.expand_dims(mask_ar.flatten(),-1))
mask_ = mask_norm.reshape(mask_ar.shape)

# separating training, validation and test
print("Separating Training, Validation and Test Sets..")

ct_valid = [ct_[-2]]; ct_test = [ct_[-1]]
pct_valid = [pct_[-2]]; pct_test = [pct_[-1]]
t1_valid = [t1_[-2]]; t1_test = [t1_[-1]]
mask_valid = [mask_[-2]]; mask_test = [mask_[-1]]

ct_train = ct_[:opt.n_subjects]; pct_train = pct_[:opt.n_subjects]
t1_train = t1_[:opt.n_subjects]; mask_train = mask_[:opt.n_subjects]

print("Changing Data Structure..")

ct_train = np.stack(ct_train); pct_train = np.stack(pct_train)
t1_train = np.stack(t1_train); #mask_train = np.vstack(mask_train)

ct_valid = np.stack(ct_valid); pct_valid = np.stack(pct_valid)
t1_valid = np.stack(t1_valid); mask_valid = np.stack(mask_valid)

ct_test = np.stack(ct_test); pct_test = np.stack(pct_test)
t1_test = np.stack(t1_test); mask_test = np.stack(mask_test)

# Preparing Sets

print("Preparing Training Data...")
X_train, Y_train, pct_train, M_train = utils.prepare_data(ct_train, \
    pct_train, t1_train, mask_train, opt.threshold, opt.no_mask, opt.no_sqrt, opt.mask_opt)

print("Preparing Validation Data...")
X_valid, Y_valid, pct_valid, M_valid = utils.prepare_data(ct_valid, \
    pct_valid, t1_valid, mask_valid, opt.threshold, opt.no_mask, opt.no_sqrt, opt.mask_opt)

print("Preparing Test Data...")
X_test, Y_test, pct_test, M_test = utils.prepare_data(ct_test, \
    pct_test, t1_test, mask_test, opt.threshold, opt.no_mask, opt.no_sqrt, opt.mask_opt)

# Concatenate pseudo CT with MRI if opt.set_pct is True.
if opt.set_pct:
    X_train = np.concatenate((X_train, pct_train), axis=-1)
    X_test = np.concatenate((X_test, pct_test), axis=-1)
    X_valid = np.concatenate((X_valid, pct_valid), axis=-1)


# Normalize data
print("Normalizing Data Sets...")

X_train, Y_train, mm_ct, mm_t1 = utils.normalize_dataset(X_train, Y_train, opt.set_pct, fit = True, tanh = opt.tanh)
X_valid, Y_valid = utils.normalize_dataset(X_valid, Y_valid, opt.set_pct, fit = False, mm_ct = mm_ct, mm_t1 = mm_t1)
X_test, Y_test = utils.normalize_dataset(X_test, Y_test, opt.set_pct, fit = False, mm_ct = mm_ct, mm_t1 = mm_t1)


print("Reshaping Data..")

if opt.set_pct:
    X_valid = X_valid.reshape(1, 256, 256, 256, 2) 
    X_test = X_test.reshape(1, 256, 256, 256, 2) 
else:
    X_valid = X_valid.reshape(1, 256, 256, 256, 1)
    X_test = X_test.reshape(1, 256, 256, 256, 1)

Y_valid = Y_valid.reshape(1, 256, 256, 256, 1)
Y_test = Y_test.reshape(1, 256, 256, 256, 1)
M_valid = M_valid.reshape(1, 256, 256, 256, 1)
M_test = M_test.reshape(1, 256, 256, 256, 1)


patch_shape = np.array(opt.patch_shape)



N_valid_samples = opt.ns_valid # number of samples to store in validation set
N_subjects_valid = 1 # only one subject for validation
disc_shape_valid = 1 # output shape of the discriminator. irrelevant parameter here.
transverse_only_valid = False # slices can be sagittal, coronal, transverse
flip_valid = True # random flip activated


X_valid_cropped, Y_valid_cropped, M_valid_cropped, valid_coords, _ = utils.load_samples(X_valid, \
    Y_valid, M_valid, N_subjects_valid, N_valid_samples, disc_shape_valid, transverse_only_valid, \
        flip_valid, patch_shape, pct = opt.set_pct, coords = None)

N_test_samples = opt.ns_valid
N_subjects_test = 1
disc_shape_test = 1
transverse_only_test = False
flip_test = True

X_test_cropped, Y_test_cropped, M_test_cropped, test_coords, _ = utils.load_samples(X_test, \
    Y_test, M_test, N_subjects_test, N_test_samples, disc_shape_test, transverse_only_test, \
        flip_test, patch_shape, pct = opt.set_pct, coords = None)


print("Saving Data Sets...")

patch_shape = tuple(patch_shape)

dataset_info = {"mm_t1": mm_t1, 
           "mm_ct": mm_ct, "mm_mask": mm_mask, "no_mask": opt.no_mask, 
           "no_sqrt": opt.no_sqrt, "seed": opt.seed, "set_pct": opt.set_pct, 
           "ns_valid": opt.ns_valid, "patch_shape": patch_shape, "tanh": opt.tanh}

if os.name == 'nt':
    data_path = "C:/Users/matth/Documents/Martinos Center/mrtoct/datasets/"
else:
    data_path = "/autofs/space/guerin/USneuromod/MATHIEU/mrtoct/datasets/"


if not os.path.exists(data_path + opt.dataset):
    os.mkdir(data_path + opt.dataset)

pickle.dump(dataset_info, open(data_path + opt.dataset + "/dataset_info", 'wb'))


# Files are saved in a 3D image format (5d-array) like .nii 
# In the format (n_subjects, height, width, depth, chans)
np.savez(data_path + opt.dataset + "/train", x=X_train, y=Y_train, m=M_train)
np.savez(data_path + opt.dataset + "/valid", x=X_valid, y=Y_valid, m=M_valid)
np.savez(data_path + opt.dataset + "/test", x=X_test, y=Y_test, m=M_test)

# These sets are in the format (n_samples, height, width, chans)
# They are the ones to be used for validating/testing the model
np.savez(data_path + opt.dataset + "/valid_eval", x=X_valid_cropped, y=Y_valid_cropped, m=M_valid_cropped, coords = valid_coords)
np.savez(data_path + opt.dataset + "/test_eval", x=X_test_cropped, y=Y_test_cropped, m=M_test_cropped, coords = test_coords)


print("Data Preprocessing: Done. Use: --dataset {} when running train.py".format(opt.dataset))