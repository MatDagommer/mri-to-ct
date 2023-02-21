# Matthieu Dagommer
# 27/06/2022

# DEFINITION OF DATA PREPROCESSING OPTIONS

import argparse

class PreprocessingOptions():

    def __init__(self):
        self.initialized = False
    
    def initialize(self, parser):
        parser.add_argument('--dataset', type = str, help = 'Name of the dataset')
        parser.add_argument('--n_subjects', type = int, default = 13, help = 'Number of subjects used for training')
        parser.add_argument('--no_mask', default = False, action = 'store_true', help = 'Mask the entire data set')
        parser.add_argument('--threshold', type = float, default = 0, help = 'Mask threshold')
        parser.add_argument('--no_sqrt', default = False, action = 'store_true', help = 'Does not use square root of porosity but porosity instead')
        parser.add_argument('--format', type = str, default = '2d', help = \
            'Type of patch to use: [2d, patch_2d, patch_3d, patch25d]. \n2d is 256x256x1 \n2d_32 is 32x32x1 \n2d_64 is 64x64x1 \n3d_32 is 32x32x32 \n3d_64 is 64x64x64 \n25d is 128x128x32') 
        parser.add_argument('--set_pct', default = False, action = 'store_true', help = 'Add a second channel corresponding to pseudo-CT')
        parser.add_argument('--seed', type = int, default = 0, help = 'Pseudo-random generator seed')
        parser.add_argument('--mask_opt', type = int, default = 0, help = 'Masking option. 0: MRI + CT mask (does not apply if no_mask = True), 1: mask CT only, 2: mask MRI only')
        parser.add_argument('--tanh', default = False, action = 'store_true', help = 'normalizes data between -1 and 1.')
        self.initialized = True
        return parser
    
    def gather_options(self):

        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
            opt = parser.parse_args()
        
        if opt.format == '2d':
            patch_shape = (256,256,1)
        elif opt.format == '2d_32':
            patch_shape = (32,32,1)
        elif opt.format == '2d_64':
            patch_shape = (64,64,1)
        elif opt.format == '2d_128':
            patch_shape = (128,128,1)
        elif opt.format == '3d_32':
            patch_shape = (32,32,32,1)
        elif opt.format == '3d_64':
            patch_shape = (64,64,64,1)
        elif opt.format == '25d': # To be interpreted as 2.5 dimension
            patch_shape = (128,128,32,1)

        ns_valid_2d = 473 # approximate usable number of 256x256x1 slices per subject with all 
                          # following planes:
                          # sagittal, coronal and transverse planes whose respective usable ranges 
                          # are [60 - 200], [40 - 230] and [90 - 230] 
        
        # To limit the disk usage, we store an approximately constant number of pixels 
        # for the validation and test set for any patch shape

        # 2d validation set requires storage of 473 * 256 * 256 = 30,998,528 voxels
        # we compute a cross-product to retrieve the number of samples (or patches) to include in 
        # the validation set for any patch_shape
        
        voxels_per_samples_2d = 473 * 256 * 256
        voxels_per_sample = patch_shape[0]*patch_shape[1]*patch_shape[2]

        ns_valid = int( voxels_per_samples_2d / voxels_per_sample )

        #if opt.patch25d:
        #    ns_valid = 14 # => 473 // 32 => 1) See explanation below. 2) We divide 473 by 32 
                          # so that the total number of stored voxels for the validation set 
                          # equals that of the 2D case.
        #else:
        #    ns_valid = 473 # approximate usable number of slices per subject with all following planes:
                           # sagittal, coronal and transverse planes whose respective usable ranges 
                           # are [60 - 200], [40 - 230] and [90 - 230] 

        parser.add_argument('--ns_valid', type = int, default = ns_valid, help = 'Number of samples used for validation')
        parser.add_argument('--ns_test', type = int, default = ns_valid, help = 'Number of samples used for test')
        parser.add_argument('--patch_shape', type = tuple, default = patch_shape, help = 'shape of a single patch of data (minimum unit of processed data)')

        self.parser = parser
        return parser.parse_args()