# Matthieu Dagommer
# 27/06/2022

# DEFINITION OF MAIN OPTIONS

import argparse
import os
from utils import utils
import pickle
import gc
import numpy as np

class TrainOptions():

    def __init__(self):
        self.initialized = False
    
    def initialize(self, parser):
        parser.add_argument('--dataset', type = str, help = 'Dataset used for training')
        parser.add_argument('--name', type = str, help = 'Name to give to the model')
        parser.add_argument('--model', type = str, default = 'custom', help = 'Network architecture: [custom, pix2pix, cyclegan]')
        parser.add_argument('--netG', type = str, default = 'UNet', help='Custom Network Generator: [UNet; UNet64; UNet128; ResNet9]')
        parser.add_argument('--netD', type = str, default = 'PatchGan', help = 'Custom Network Discriminator: [PatchGan; CycleGan; FilaSGan]')
        parser.add_argument('--no_bim', default = False, action = 'store_true', help = 'Remove Backpropagation In the Mask only')
        parser.add_argument('--bs', type = int, default = 20, help = 'Mini-batch size')
        parser.add_argument('--n_epochs', type = int, default = 15, help = 'Number of epochs')
        parser.add_argument('--n_subjects', type = int, default = 13, help = 'Number of subjects used for training')
        parser.add_argument('--to', default = False, action = 'store_true', help = 'transverse slices only')
        parser.add_argument('--no_flip', default = False, action = 'store_true', help = 'If False, data augmentation with flipped images')
        parser.add_argument('--gdl', default = False, action = 'store_true', help = 'Adds Gradient-Difference Loss to train the cGAN.')
        parser.add_argument('--l1_l2', default = False, action = 'store_true', help = 'Adds both L1 and L2 losses to train the cGAN.')
        parser.add_argument('--save_freq', type = int, default = 10, help='frequency of saving training/validation and model results (in batches)')
        parser.add_argument('--gdl_metric', type = str, default = 'mse', help = 'Metric used for Gradient-Difference')
        parser.add_argument('--loss_metric', type = str, default = 'mae', help = 'Main loss metric: mae, mse, mpd. If l1_l2, metric becomes [mae, mse].')
        parser.add_argument('--acm', default = False, action = 'store_true', help = 'Name of previous generator to use for Auto-Context Model')
        parser.add_argument('--feature_loss', default = False, action = 'store_true', help = 'Add content+style losses with pretrained VGG16 model')
        parser.add_argument('--perceptual_loss', default = False, action = 'store_true', help = 'Add perceptual loss at the discriminator-level.')
        #parser.add_argument('--tanh', default = False, action = 'store_true', help = 'normalizes data between -1 and 1.')
        parser.add_argument('--discls', default = False, action = 'store_true', help = 'discriminator least squares. Replaces binary cross-entropy with least square loss functions.')
        self.initialized = True
        return parser
    
    def gather_options(self):

        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        
        # Parse to retrieve opt.dataset
        opt, _ = parser.parse_known_args()

        # Retrieve dataset-specific options
        parser = utils.RetrieveDataSetOptions(parser, opt.dataset)
        opt = parser.parse_args()


        # Retrieve Model and Data Paths
        if os.name == 'nt': # Windows (my personal computer)
            models_dir = "C:/Users/matth/Documents/Martinos Center/mrtoct/checkpoints/"
            if opt.acm:
                acm_list = [i for i in os.listdir(models_dir) if opt.name + "_acm" in i]
                num_acm = len(acm_list) + 1
                opt.name = opt.name + "_acm{}".format(num_acm)
                #model_path = "C:/Users/matth/Documents/Martinos Center/mrtoct/checkpoints/" + opt.name + "_acm{}/".format(num_acm)

            model_path = "C:/Users/matth/Documents/Martinos Center/mrtoct/checkpoints/" + opt.name + "/"
            data_path = "C:/Users/matth/Documents/Martinos Center/mrtoct/datasets/"
        else: # CentOS7 (Deepbrain)
            models_dir = "/autofs/space/guerin/USneuromod/MATHIEU/mrtoct/checkpoints/"
            if opt.acm:
                acm_list = [i for i in os.listdir(models_dir) if opt.name + "_acm" in i]
                num_acm = len(acm_list) + 1
                opt.name = opt.name + "_acm{}".format(num_acm)
            #model_path = "C:/Users/matth/Documents/Martinos Center/mrtoct/checkpoints/" + opt.name + "_acm{}/".format(num_acm)

            model_path = "/autofs/space/guerin/USneuromod/MATHIEU/mrtoct/checkpoints/" + opt.name + "/"
            data_path = "/autofs/space/guerin/USneuromod/MATHIEU/mrtoct/datasets/"

        
        parser.add_argument('--models_dir', type = str, default = models_dir, help = 'Directory containing saved models')
        parser.add_argument('--model_path', type = str, default = model_path, help = 'Where the current model is saved')
        parser.add_argument('--data_path', type = str, default = data_path, help = 'Where Data is saved')

        
        
        # Define Number of Samples per Epoch
        sag = 141; cor = 191; tra = 141 
        # sagittal, coronal, transversal => See preprocessing.py
        # Sum of all these slices is close to 500
        if opt.netG == "UNet128":
            ns_per_epo = opt.n_subjects * (sag + cor + tra) * int((256 / 128) ** 2)
        elif opt.netG == "UNet64":
            ns_per_epo = opt.n_subjects * (sag + cor + tra) * int((256 / 64) ** 2)
        else:
            ns_per_epo = opt.n_subjects * (sag + cor + tra)
        print("Number of samples per epoch: ", ns_per_epo)

        # Set the number of samples per epoch
        bat_per_epo = int(ns_per_epo / opt.bs) 
        ns_per_epo = bat_per_epo * opt.bs
        save_every_x_batches = bat_per_epo // opt.save_freq


        parser.add_argument('--ns_per_epo', type = int, default = ns_per_epo, help = 'Number of Samples per Epoch')
        parser.add_argument('--bat_per_epo', type = int, default = bat_per_epo, help = "Number of batches per epo")
        parser.add_argument('--save_every_x_batches', type = int, default = save_every_x_batches, help = 'Number of batches between each saving')

        # Input Shape
        
        shape = list(opt.patch_shape)
        print('type of shape: ', type(shape))
        if opt.set_pct:
            if opt.acm:
                shape[-1] *= 3
                # If we add acm above pseudo-ct, we concatenate t1, pCT and previous output (3 layers)
            else:
                shape[-1] *= 2
        else:
            if opt.acm:
                shape[-1] *= 2
            else:
                pass
        input_shape = tuple(shape)

        # Output Shape

        output_shape = opt.patch_shape
        if opt.set_pct or opt.acm:
            temp = list(output_shape)
            temp[-1] = 1
            output_shape = tuple(temp)
        print('output_shape: ', output_shape)

        # Auto-Context Model
        # Concatenation of source image + previous model output => # channels doubles
    

        parser.add_argument('--input_shape', type = tuple, default = input_shape, help = 'Input Data Shape (MRI Samples).')
        parser.add_argument('--output_shape', type = tuple, default = output_shape, help = 'Output Data Shape (CT Samples).')
        
        self.parser = parser
        opt_ = parser.parse_args()
        if opt.acm:
            opt_.name = opt_.name + "_acm{}".format(num_acm)

        # Remove bim when testing classical networks
        if opt.model in ['pixtopix', 'cyclegan']:
            opt_.no_bim = True
        
        return opt_

if __name__ == '__main__':
    opt = TrainOptions().gather_options()
    pickle.dump(opt, open(opt.data_path + "exps/" + opt.name, "wb"))
        