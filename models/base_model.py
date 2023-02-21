# Matthieu Dagommer
# 04/07/2022
import utils.utils as utils
import numpy as np
from abc import abstractmethod
#import networks

class BaseModel():
    # Abstract Base Class to define common properties/methods of Models.

    def __init__(self, opt):
        # opt => combined data / training options
        self.opt = opt

        # Build the GAN
    
    @abstractmethod
    def train_on_batch():
        pass


    #@staticmethod
    def generate_fake_samples(self, input, disc_shape):# acm_model = None):

        output = self.batch_computation(input)
        disc_y_shape = tuple([len(output)] + list(disc_shape))
        y_fake = np.zeros(shape=disc_y_shape)
        return output, y_fake


    #@staticmethod
    def compute_train_error(self, opt, real, fake, mask):

        # normalize back from [-1;1] to [0;1]
        if opt.tanh:
            real, fake = (real + 1) / 2, (fake + 1) / 2 

        if not opt.no_sqrt:
            diff = np.square(fake) - np.square(real)
        else:
            diff = np.array(fake) - np.array(real)
        errors_flat = diff.flatten()
        mask_flat = mask.flatten()
        n_mask_pixels = int(mask_flat.sum())
        train_error = np.abs(errors_flat[np.where(mask_flat == 1)]).sum()
        return train_error, n_mask_pixels

    #@staticmethod
    def compute_valid_error(self, opt, source, target, mask):

        #print('source: ', source.shape )
        generated = self.batch_computation(source)
        #print('generated: ', generated.shape)

        # normalize back from [-1;1] to [0;1]
        if opt.tanh:
            generated, target = (generated + 1) / 2, (target + 1) / 2 
        
        if not opt.no_sqrt:
            valid_errors_matrix = np.square(generated) - np.square(target)
        else:
            valid_errors_matrix = generated - target
        #print('valid_error_matrix: ', valid_errors_matrix.shape)
        mask_flat = mask.flatten()
        n_mask_pixels = int(mask_flat.sum())
        valid_errors_flat = valid_errors_matrix.flatten()
        val_error = (1/(n_mask_pixels))*np.abs(valid_errors_flat[np.where(mask_flat == 1)]).sum()
        return generated, valid_errors_flat, val_error

    #@staticmethod
    def save_model(self, opt):
        self.generator.save(opt.model_path + opt.name + "_g.h5")
        self.gan.save(opt.model_path + opt.name + "_gan.h5")
        self.discriminator.save(opt.model_path + opt.name + "_d.h5")

    #@staticmethod
    def get_summary(self, opt):
        with open(opt.model_path + 'gan_summary.txt','w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.gan.summary(print_fn=lambda x: fh.write(x + '\n'))
        with open(opt.model_path + 'g_summary.txt','w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.generator.summary(print_fn=lambda x: fh.write(x + '\n'))
        with open(opt.model_path + 'd_summary.txt','w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.discriminator.summary(print_fn=lambda x: fh.write(x + '\n'))
        ### Plot Graph of the Model
        #keras.utils.plot_model(gan_model, "C:/Users/matth/Documents/Martinos Center/Rapport/cGAN_model_2.png", show_shapes=True)
    
    #@staticmethod
    def generate_images(self, test_set, mm_ct):
        utils.generate_images(self.generator, test_set, mm_ct)

    #@staticmethod
    def discriminator_output_shape(self):
        return self.discriminator.output_shape[-2]

    #@staticmethod
    def batch_computation(self, input):
        temp_bs = 200 # temporary batch size
        q = input.shape[0] // temp_bs
        r = input.shape[0] % temp_bs
        nb_iters = q if r == 0 else q + 1
        generated = []
        for i in range(nb_iters):
            if i < nb_iters - 1:
                generated.append(self.generator.predict(input[temp_bs*i:temp_bs*(i+1)], verbose = 0))
            else:
                generated.append(self.generator.predict(input[temp_bs*i:], verbose = 0))
        generated = np.concatenate(generated, axis = 0)
        return generated