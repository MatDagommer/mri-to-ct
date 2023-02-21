## Matthieu Dagommer
# 27/06/2022

# MAIN TRAINING ALGORITHM

import os
import numpy as np
import gc
import pickle
import utils.utils as utils
import tensorflow as tf
import models.networks as models
from keras.models import load_model
import time

def train_function(model, train_set, valid_set, opt):
#def train_function(model, opt):

    # -------------------- Retrieve Parameters --------------------- #


    print("Retrieving Parameters..")
    if not os.path.exists(opt.model_path):
        os.mkdir(opt.model_path)

    # determine the output square shape of the discriminator:
    #n_patch = model.discriminator_output_shape()
    if len(opt.patch_shape) == 4: # 3D-patches
        n_patch = model.discriminator.output_shape[-4:]
    else:
        n_patch = model.discriminator.output_shape[-3:]
    #print("==> discriminator output shape: ", n_patch)
    if opt.perceptual_loss:
        n_patch = model.discriminator.output_shape[0][-3:]
    
    print('n_patch', n_patch)
    
    trainA, trainB, M_train = train_set["x"], train_set["y"], train_set["m"]
    valA, valB, valM = valid_set["x"], valid_set["y"], valid_set["m"]

    #print('valA: ', valA.shape, 'valB: ', valB.shape, 'valM: ', valM.shape)

    # Auto-Context Model
    #if opt.acm is not None:
    if opt.acm:
        print("Adding Auto-Context Model..")

        h5_files = []
        original_model = opt.name; original_model = original_model[:-5]
        h5_files.append(original_model)
        for i in os.listdir(opt.models_dir):
            if original_model + "_acm" in i:
                h5_files.append(i)
        h5_files.sort()
        h5_files = h5_files[:-1] # We don't include the last model, which is about to be trained
        acm_models = []

        for i in h5_files:
            acm_model = load_model(opt.models_dir + i + '/' + i + '_g.h5')
            acm_models.append(acm_model)
        
        valid_acm = acm_models[0].predict(valA)
        for i in range(1, len(acm_models)):
            if len(acm_models) > 2:
                valid_acm = np.concatenate((valA, valid_acm), axis = -1)
                valid_acm = acm_models[i].predict(valid_acm)
                
        
        valA = np.concatenate((valA, valid_acm), axis = -1)
        """
        subA_acm = acm_model.predict(subA)
        subA = np.concatenate((subA, subA_acm), axis = -1)"""
        print("number of acm_models: ", len(acm_models))
    else:
        acm_model = None
    
    
    
    # Masks
    M_valid_flat = valM.flatten()
    #N_mask_pixels_valid = int(M_valid_flat.sum())
    
    
    # Number of batches per training epoch
    bat_per_epo = opt.bat_per_epo
    
    valid_errors, train_errors = [], []
    bevel, begen, begt = [], [], [] 

    # Best Epoch: -Validation Error List, -Generated Porosity List, -Ground-Truth Porosity List
    min_validation_MAE = 100000

    # Initialize train_error count => Retrieve errors
    train_error = 0
    train_voxels_count = 0
    

    # -------------------- Training -------------------- #

    print("Training..") 
    for epoch in range(opt.n_epochs + 1):

        for batch in range(bat_per_epo):
            
            # Compute a single iteration for epoch 0
            if epoch == 0 and batch > 0:
                break
            # Select a batch of real samples
            X_realA, X_realB, coord, y_real = utils.generate_real_samples(trainA, trainB, opt.bs, \
                n_patch, opt.n_subjects, opt.to, opt.no_flip, opt.set_pct, opt.patch_shape)

            #X_realA, X_realB, mask_batch, y_real = utils.load_samples(opt, trainA, trainB, M_train, opt.n_subjects, opt.bs, n_patch)
            # Retrieve Masks that correspond with the inputs
            #mask_batch = models.retrieve_masks(opt, coord, M_train, X_realA.shape)
            
            mask_batch = utils.retrieve_slices(coord, opt.bs, M_train, opt.output_shape)

            # Auto-Context Model
            if opt.acm:
                X_acm = acm_models[0].predict(X_realA)
                
                if len(acm_models) > 1:
                    for i in range(1, len(acm_models)):
                            X_acm = np.concatenate((X_realA, X_acm), axis = -1)
                            X_acm = acm_models[i].predict(X_acm)

                X_realA = np.concatenate((X_realA, X_acm), axis = -1)

            # Generate predictions ("fakes") with the model
            X_fakeB, y_fake = model.generate_fake_samples(X_realA, n_patch)

            # Apply Mask to generated CT if masked_input
            if not opt.no_bim:
                #print("Warning: bim is on in train_function.")
                X_fakeB = tf.math.multiply(X_fakeB - 1, mask_batch) + 1
                X_fakeB = tf.cast(X_fakeB, dtype = tf.float32)


            if epoch > 0:
                # Train Models
                model.train_on_batch(opt, X_realA, X_realB, X_fakeB, mask_batch, y_real, y_fake)
            
            # Update error count
            te, tv = model.compute_train_error(opt, X_realB, X_fakeB, mask_batch)
            train_error += te
            train_voxels_count += tv

            # Save results <opt.save_freq> times per epoch (including last step)
            if epoch > 0 and (((batch+1) % opt.save_every_x_batches == 0 \
                and (batch+1) < opt.save_every_x_batches * opt.save_freq) \
                or (batch+1) == opt.bat_per_epo) or epoch == 0:
                
                # Compute validation error (on validation set)
                Y_valid, valid_errors_flat, val_error = model.compute_valid_error(opt, valA, valB, valM)
                valid_errors.append(val_error)

                # Retrieve training error and reset
                train_errors.append(train_error/train_voxels_count)
                train_error = 0
                train_voxels_count = 0

                # Save Model if it's the Best Model so far
                if valid_errors[-1] < min_validation_MAE:

                    min_validation_MAE = valid_errors[-1]
                    bevel = valid_errors_flat[np.where(M_valid_flat == 1)].tolist()
                    begen = Y_valid.flatten()[np.where(M_valid_flat == 1)].tolist()

                    if opt.tanh:
                        valB_ = ( valB + 1 ) / 2
                    else:
                        valB_ = valB.copy()
                        
                    begt = valB_.flatten()[np.where(M_valid_flat == 1)].tolist()
                    
                    model.save_model(opt)

                # Summarize performance (at each every <display_iters> batches)
                if epoch == 0:
                    print('>Epoch %d: train error = %.6f; valid error = %.6f.' % (epoch, train_errors[-1], valid_errors[-1]))
                else:
                    print('>Epoch %d, (#batch: %d / %d): train error = %.6f; valid error = %.6f.' % (epoch, batch+1, \
                                                                                            bat_per_epo, train_errors[-1], valid_errors[-1]))
                
                # Save temporary output
                pickle.dump([train_errors, valid_errors, bevel, begen, begt], open(opt.model_path + "temp_output", "wb"))

    return [train_errors, valid_errors, bevel, begen, begt]