# Matthieu - 01/01/2023

import sys
sys.path.append("C:/Users/matth/Documents/Martinos Center/mrtoct") 

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import os
import pickle
import numpy as np
from models.networks import ReflectionPadding2D, ReflectionPadding3D
import scipy
import scipy.stats as stats
from scipy.stats import kurtosis, skew
from sklearn.metrics import r2_score
from tqdm import tqdm
from sklearn.metrics import r2_score
from skimage.metrics import structural_similarity
import pandas as pd


def retrieve_models(model_names, fpath):
    models = []
    outputs = []
    for name in model_names:
        path = os.path.join(fpath, name)
        model = load_model(path + '/' + name + '_g.h5', compile = False, custom_objects = {"ReflectionPadding2D": ReflectionPadding2D, "ReflectionPadding3D": ReflectionPadding3D})
        output = pickle.load(open(path + '/' + 'output_' + name, 'rb'))
        models.append(model)
        outputs.append(output)
    return models, outputs

def retrieve_history(outputs):
    trains, vals, bevels, begens, begts = [], [], [], [], []
    for output in outputs:
        train, val, bevel, begen, begt = output
        trains.append(train)
        vals.append(val)
        bevels.append(bevel)
        begens.append(begen)
        begts.append(begt)
    return trains, vals, bevels, begens, begts

def compute_errors(dataset, model = None, pct = False):
    if pct: 
        y_pred = dataset['x'][:,:,:,1:2]
    else:
        x = dataset['x']
        #print(x.shape)
        y_pred = np.zeros_like(x)
        for i in range(x.shape[0]):
            y_pred[i:i+1] = model.predict(x[i:i+1])

    y = dataset['y']
    m = dataset['m']
    
    y_pred = ( y_pred + 1 ) / 2
    y = ( y + 1 ) / 2

    begens = list(y_pred.flatten()[np.where(m.flatten() == 1)]) 
    begts = list(y.flatten()[np.where(m.flatten() == 1)]) 
    bevels = list(np.array(begens) - np.array(begts)) 

    return [begens], [begts], [bevels]

def compute_metrics(model_name, bevel, begen, begt, dataset, test = False):
    N = len(bevel) # number of voxels in validation set
    print("N = ", N)
    L = 1
    bevel = np.array(bevel)

    mask = dataset["m"]
    print("sum mask: ", int( mask.flatten().sum() ))
    #print('shape 1', mask.shape)
    if test:
        mask = mask[0]
        #print('shape 2', mask.shape)

    MAE = 1 / N * np.sum(np.abs(bevel))
    MSE = 1 / N * np.sum(np.square(bevel))
    M = np.max( begen + begt ) - np.min( begen + begt )
    PSNR = 10 * np.log( M**2 / MSE )

    mae_list, mse_list, psnr_list = [], [], []

    count = 0
    #print('mask shape before computing stddv: ', mask.shape[0])
    for i in range(mask.shape[0]):
        count_ = int( mask[i].flatten().sum() )
        if count_ == 0:
            pass
        else:
            mae_list.append( np.abs(bevel[count:count + count_]).flatten().mean() )
            mse_list.append( np.square(bevel[count:count + count_]).flatten().mean() )
            psnr_list.append( 10* np.log( M**2 / mse_list[-1]) )
        count += count_
    
    MAE_sd = np.std(mae_list)
    MSE_sd = np.std(mse_list)
    PSNR_sd = np.std(psnr_list)
    
    return MAE, MAE_sd, MSE, MSE_sd, PSNR, PSNR_sd

# Compute SSIMs
def compute_ssim(model, dataset, acm = False, sqrt = False, test = False, Y_pred = None, pct = False):
    X, Y, M = dataset['x'], dataset['y'], dataset['m']

    if test:
        ssims = []
        if pct:
            Y_pred = X[:,:,:,1:2]
        else:
            X = X[0]
            Y = Y[0]
            Y_pred = Y_pred[0]
            M = M[0]

        for i in range(Y.shape[0]):
            if len(Y.shape) == 5:
                ssim = structural_similarity(Y[i,:,:,32,0], Y_pred[i,:,:,32,0])
            else:
                ssim = structural_similarity(Y[i,:,:,0], Y_pred[i,:,:,0])
            ssims.append(ssim)

        ssim_average = np.mean(ssims)
        sd = np.std(ssims)

        return ssim_average, sd
    
    N_samples = X.shape[0]
    if not acm:
        ssim_sum = 0
        ssims = []
        for i in range(N_samples):
            Y_gen = model.predict(X[i:i+1])
            Y_gen = np.multiply(Y_gen - 1, M[i:i+1]) + 1
            #print(Y_gen.shape)
            Y_mask = np.multiply(Y[i:i+1] - 1, M[i:i+1]) + 1
            if sqrt:
                #ssim = structural_similarity(np.square(Y_gen[0,:,:,0]), np.square(Y_mask[0,:,:,0]))
                if len(Y.shape) == 5:
                    ssim = structural_similarity(np.square(Y[i,:,:,32,0]), np.square(Y_gen[0,:,:,32,0]))
                else:
                    ssim = structural_similarity(np.square(Y[i,:,:,0]), np.square(Y_gen[0,:,:,0]))
            else:
                #ssim = structural_similarity(Y_gen[0,:,:,0], Y_mask[0,:,:,0])
                if len(Y.shape) == 5:
                    ssim = structural_similarity(Y[i,:,:,32,0], Y_gen[0,:,:,32,0])
                else:
                    ssim = structural_similarity(Y[i,:,:,0], Y_gen[0,:,:,0])
            ssim_sum += ssim
            ssims.append(ssim)
        ssim_sum /= N_samples
        sd = np.std(ssims)
    else:
        ssim_sum = np.zeros((3))
        ssims_1, ssims_2, ssims_3, = [], [], []
        ssims = [ssims_1, ssims_2, ssims_3]
        for i in tqdm(range(N_samples)):
            Y_gen = model[0].predict(X[i:i+1])
            Y_mask = np.multiply(Y[i:i+1] - 1, M[i:i+1]) + 1
            for j in range(1, len(model)):
                Y_gen = model[j].predict( np.concatenate((X[i:i+1], Y_gen), axis = -1) )
                Y_gen_mask = np.multiply(Y_gen - 1, M[i:i+1]) + 1
                if sqrt:
                    ssim = structural_similarity(np.square(Y_gen_mask[0,:,:,0]), np.square(Y_mask[0,:,:,0]))
                else:
                    ssim = structural_similarity(Y_gen_mask[0,:,:,0], Y_mask[0,:,:,0])
                ssim_sum[j-1] += ssim
                ssims[j-1].append(ssim)
        ssim_sum = ssim_sum / N_samples
        sd_1 = np.std(ssims[0])
        sd_2 = np.std(ssims[1])
        sd_3 = np.std(ssims[2])
        sd = [sd_1, sd_2, sd_3]

    return ssim_sum, sd

def compute_val_metrics(models, model_names, datasets, bevels, begens, begts, sqrt = False, acm = False, test = False, pct = False):
    MAE, MAE_sd, MSE, MSE_sd, PSNR, PSNR_sd, SSIM, SSIM_sd = [], [], [], [], [], [], [], []
    for i in tqdm(range(len(model_names))):
        if type(datasets) == list:
            dataset = datasets[i]
        else:   
            dataset = datasets
        if pct:
            mae, mae_sd, mse, mse_sd, psnr, psnr_sd = compute_metrics(model_names, bevels, begens, begts, dataset)
        else:
            mae, mae_sd, mse, mse_sd, psnr, psnr_sd = compute_metrics(model_names[i], bevels[i], begens[i], begts[i], dataset)

        MAE.append(mae)
        MAE_sd.append(mae_sd)
        MSE.append(mse)
        MSE_sd.append(mse_sd)
        PSNR.append(psnr)
        PSNR_sd.append(psnr_sd)

        if not acm:
            ssim, ssim_sd = compute_ssim(models[i], dataset, sqrt = sqrt, test = test, pct = pct)
            SSIM.append(ssim)
            SSIM_sd.append(ssim_sd)
    if acm:
        SSIM, SSIM_sd = compute_ssim(models, datasets, sqrt = sqrt, acm = True)
        df = pd.DataFrame({' ': model_names[1:], 'MAE': MAE[1:], '$\sigma_{MAE}$': MAE_sd[1:], 'MSE': MSE[1:], '$\sigma_{MSE}$': MSE_sd[1:], 'PSNR': PSNR[1:], '$\sigma_{PSNR}$': PSNR_sd[1:], 'SSIM': SSIM, '$\sigma_{SSIM}$': SSIM_sd}).set_index(' ')
        pd.set_option('display.float_format', '{:.4f}'.format)
    else:
        df = pd.DataFrame({' ': model_names, 'MAE': MAE, '$\sigma_{MAE}$': MAE_sd, 'MSE': MSE, '$\sigma_{MSE}$': MSE_sd, 'PSNR': PSNR, '$\sigma_{PSNR}$': PSNR_sd, 'SSIM': SSIM, '$\sigma_{SSIM}$': SSIM_sd}).set_index(' ')
        pd.set_option('display.float_format', '{:.4f}'.format)
    return df


def compute_test_metrics(model, model_name, dataset, recon_3d = False, width_2d = 256, sqrt = False, pct = False):
    
    X, Y, M = dataset['x'], dataset['y'], dataset['m']
    if sqrt:
        Y = np.square(Y)

    if pct:
        
        gen_vol = X[:,:,:,:,1:2]
        Y = ( Y + 1 ) / 2
        gen_vol = ( gen_vol + 1 ) / 2

        begen = list(gen_vol.flatten()[np.where(M.flatten() == 1)])
        begt = list(Y.flatten()[np.where(M.flatten() == 1)])
        bevel = list(np.array(begen) - np.array(begt))   

        mae, mae_sd, mse, mse_sd, psnr, psnr_sd = compute_metrics(model_name, bevel, begen, begt, dataset, test = True)
        ssim, ssim_sd = compute_ssim(model, dataset, sqrt = sqrt, test = True, Y_pred = gen_vol)

    elif recon_3d:
        
        temp_vol = np.zeros((256,256,256,1))
        nb_sums = np.zeros((256,256,256,1))

        blocks_per_axis = int((256 - 64) / 32 + 1)
        s = 32 # side of a block

        for i in tqdm(range(blocks_per_axis)):
            for j in range(blocks_per_axis):
                for k in range(blocks_per_axis):
                    block = model.predict(X[:,i*s:(i+2)*s,j*s:(j+2)*s,k*s:(k+2)*s])
                    if sqrt:
                        block = np.square(block)
                    temp_vol[i*s:(i+2)*s,j*s:(j+2)*s,k*s:(k+2)*s] += block[0]
                    nb_sums[i*s:(i+2)*s,j*s:(j+2)*s,k*s:(k+2)*s] += 1
        
        averaged_vol = np.divide(temp_vol, nb_sums)
        gen_vol = np.multiply(averaged_vol - 1, M[0]) + 1

        # TANH
        gen_vol = (gen_vol + 1) / 2 
        Y = (Y + 1) / 2
        
        begen = list(gen_vol.flatten()[np.where(M[0].flatten() == 1)])
        begt = list(Y[0].flatten()[np.where(M[0].flatten() == 1)])
        bevel = list(np.array(begen) - np.array(begt))

        gen_vol = np.reshape(gen_vol, (1, 256, 256, 256, 1))
        mae, mae_sd, mse, mse_sd, psnr, psnr_sd = compute_metrics(model_name, bevel, begen, begt, dataset, test = True)
        ssim, ssim_sd = compute_ssim(model, dataset, sqrt = sqrt, test = True, Y_pred = gen_vol)
    
    else:
        S_slices = np.zeros((256,256,256,1))
        C_slices = np.zeros((256,256,256,1))
        T_slices = np.zeros((256,256,256,1))
        nb_sums = np.zeros((256,256,256,1))

        # 128
        l = width_2d # width of crop
        s = int(l / 2) # stride
        N_crops_per_dir = int(256 / s) - 1

        for i in tqdm(range(256)): # volume is 256x256x256
            for j in range(N_crops_per_dir):
                for k in range(N_crops_per_dir):
                    S_slices[i, j*s:j*s+l, k*s:k*s+l, :] += model.predict( X[:, j*s:j*s+l, i, k*s:k*s+l, :] )[0]
                    C_slices[i, j*s:j*s+l, k*s:k*s+l, :] += model.predict( X[:, j*s:j*s+l, i, k*s:k*s+l, :] )[0] 
                    T_slices[i, j*s:j*s+l, k*s:k*s+l, :] += model.predict( X[:, j*s:j*s+l, k*s:k*s+l, i, :] )[0]
                    nb_sums[i, j*s:j*s+l, k*s:k*s+l, :] += 1
        
        S_slices_ = S_slices / nb_sums
        C_slices_ = C_slices / nb_sums
        T_slices_ = T_slices / nb_sums

        S_vol = S_slices_.copy()
        C_vol = C_slices_.transpose([0, 2, 1, 3, 4])
        T_vol = T_slices_.transpose([0, 2, 3, 1, 4])

        averaged_vol = 1 / 3 * (S_vol + C_vol + T_vol)
        
        gen_vol = np.multiply(averaged_vol - 1, M[0]) + 1

        # TANH
        gen_vol = (gen_vol + 1) / 2 
        Y = (Y + 1) / 2

        begen = list(gen_vol.flatten()[np.where(M[0].flatten() == 1)])
        begt = list(Y.flatten()[np.where(M[0].flatten() == 1)])
        bevel = list(np.array(begen) - np.array(begt))   

        gen_vol = np.reshape(gen_vol, (1,256,256,256,1))
        mae, mae_sd, mse, mse_sd, psnr, psnr_sd = compute_metrics(model_name, bevel, begen, begt, dataset, test = True)
        ssim, ssim_sd = compute_ssim(model, dataset, sqrt = sqrt, test = True, Y_pred = gen_vol)
    
    df = pd.DataFrame({' ': [model_name], 'MAE': [mae], '$\sigma_{MAE}$': [mae_sd], 'MSE': [mse], '$\sigma_{MSE}$': [mse_sd], 'PSNR': [psnr], '$\sigma_{PSNR}$': [psnr_sd], 'SSIM': [ssim], '$\sigma_{SSIM}$': [ssim_sd]}).set_index(' ')
    pd.set_option('display.float_format', '{:.4f}'.format)

    return df, gen_vol #, MAE_slices


def compute_maes(model, dataset):
    
    x, y, m = dataset['x'], dataset['y'], dataset['m']
    y_pred = np.zeros_like(y)
    
    for i in range(x.shape[0]):
        y_pred[i:i+1] = model.predict(x[i:i+1])
    
    y_pred = ( y_pred + 1 ) / 2
    y = ( y + 1 ) / 2

    begen = list(y_pred.flatten()[np.where(m.flatten() == 1)])
    begt = list(y.flatten()[np.where(m.flatten() == 1)])
    bevel = list(np.array(begen) - np.array(begt))

    bevel = np.array(bevel)
    maes_list = []
    count = 0
    for i in range(x.shape[0]):
        count_ = int( m[i].flatten().sum() )
        if count_ == 0:
            pass
        else:
            maes_list.append( np.abs(bevel[count:count + count_]).flatten().mean() )
        count += count_
    
    MAE_sd = np.std(maes_list)

    return maes_list, MAE_sd