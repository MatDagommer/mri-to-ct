# Training Using ID-CGAN

from models.base_model import BaseModel
import models.networks as networks
from tensorflow import keras
import numpy as np
import models.networks as networks
#from models.networks import vgg16_preprocess
import utils.utils as utils
import time
import tensorflow as tf

from keras.models import Model, Sequential
from keras import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate, Add
from keras.layers import Dropout
from keras.layers import BatchNormalization
from tensorflow_addons.layers import InstanceNormalization
from keras.layers import Layer
from keras.layers import InputSpec
from keras.layers import ZeroPadding2D
from keras.layers import Cropping2D
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Flatten
from keras.applications.vgg16 import VGG16

class IDcGAN(BaseModel):

    def __init__(self, opt):

        ngf = 64
        ndf = 48

        super(IDcGAN, self).__init__(opt)
        input_nc = 1
        output_nc = 1
        self.vgg16 = None
        #def generator_containing_discriminator(generator, discriminator):
        self.generator = networks.create_idcgan_generator(input_nc, output_nc, ngf)
        self.discriminator = networks.create_idcgan_discriminator(input_nc, output_nc, ndf, 3)

        inputs = Input(shape=(256,256,1))
        mask_src = Input(shape = opt.output_shape) # Adding mask as an input 
        x_generator = self.generator(inputs)

        if not opt.no_bim:
            #print("Warning: bim is on in custom_model.")
            x_generator = tf.math.multiply(x_generator - 1, mask_src) + 1
            tf.cast(x_generator, dtype = tf.float32)
        
        merged = Concatenate(axis=3)([inputs, x_generator])
        self.discriminator.trainable = False
        x_discriminator = self.discriminator(merged)

        concatenated = Concatenate(axis=0)([inputs, x_generator])
        base_model = VGG16(weights='imagenet', include_top=False)


        vgg_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_conv2').output)
        vgg_model.trainable = False

        concatenated_vgg = networks.vgg16_preprocess(concatenated[:,:,:,0])
        vgg_model_out = vgg_model(concatenated_vgg)
        
        self.gan = Model(inputs=[inputs, mask_src], outputs=[x_generator,x_discriminator,vgg_model_out])
    
        #return model, x_generator, x_discriminator, vgg_model_out

        g_optim = keras.optimizers.Adam(learning_rate=0.002,beta_1=0.5)
        d_optim = keras.optimizers.Adam(learning_rate=0.002,beta_1=0.5)

        self.discriminator.compile(d_optim, loss=networks.idcgan_discriminator_loss)
        self.generator.compile(g_optim, loss='mse')
        self.gan.compile(g_optim, loss = [networks.idcgan_refined_loss(d_out=x_discriminator, \
            vgg_out=vgg_model_out),networks.idcgan_constant_loss,networks.idcgan_constant_loss])

    def generate_fake_samples(self, input, disc_shape):# acm_model = None):

        output = self.batch_computation(input)
        disc_y_shape = tuple([len(output)] + list(disc_shape))
        y_fake = np.zeros(shape=disc_y_shape)
        return output, y_fake


    def train_on_batch(self, opt, X_realA, X_realB, X_fakeB, mask_batch, y_real, y_fake):
        # Select Targets that Match Model Outputs
        outs = networks.retrieve_outputs(opt, X_realB, mask_batch, y_real, self.vgg16, X_realA)
        # Train Models
        real_pairs = np.concatenate((X_realA, X_realB), axis=3)
        fake_pairs = np.concatenate((X_realA, X_fakeB), axis=3)
        self.discriminator.trainable = True
        #_ = self.discriminator.train_on_batch([X_realA, X_realB], y_real)
        #_ = self.discriminator.train_on_batch([X_realA, X_fakeB], y_fake)
        x = np.concatenate((real_pairs, fake_pairs))
        y = np.concatenate((np.ones((opt.bs, 32, 32, 1)),np.zeros((opt.bs, 32, 32, 1))))
        _ = self.discriminator.train_on_batch(x, y)

        self.discriminator.trainable = False
        #_ = self.gan.train_on_batch([X_realA, mask_batch], outs)
        rand = np.ones((opt.bs, 32, 32, 1))
        _ = self.gan.train_on_batch([X_realA, mask_batch], [X_realB,rand,rand])
        self.discriminator.trainable = True

    def compute_train_error(self, opt, real, fake, mask):
        if not opt.no_sqrt:
            diff = np.square(fake) - np.square(real)
        else:
            diff = np.array(fake) - np.array(real)
        errors_flat = diff.flatten()
        mask_flat = mask.flatten()
        n_mask_pixels = int(mask_flat.sum())
        train_error = np.abs(errors_flat[np.where(mask_flat == 1)]).sum()
        return train_error, n_mask_pixels

    def compute_valid_error(self, opt, source, target, mask):

        #print('source: ', source.shape )
        generated = self.batch_computation(source)
        #print('generated: ', generated.shape)
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

    def save_model(self, opt):
        self.generator.save(opt.model_path + opt.name + "_g.h5")
        self.gan.save(opt.model_path + opt.name + "_gan.h5")
        self.discriminator.save(opt.model_path + opt.name + "_d.h5")

    def get_summary(self, opt):
        with open(opt.model_path + 'gan_summary.txt','w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.gan.summary(print_fn=lambda x: fh.write(x + '\n'))

    def generate_images(self, test_set, mm_ct):
        utils.generate_images(self.generator, test_set, mm_ct)

    def discriminator_output_shape(self):
        return self.discriminator.output_shape[-2]


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