# Matthieu Dagommer
# 04/07/2022

# Custom Class to define our own architecture

from models.base_model import BaseModel
import models.networks as models
from models.networks import gram_matrix
import utils.utils as utils
import tensorflow as tf
from tensorflow import keras
from keras.layers import BatchNormalization, Input
from keras.models import Model
import numpy as np

class PixToPix(BaseModel):

    def __init__(self, opt):
        
        super(PixToPix, self).__init__(opt)
        optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.generator = models.define_generator(opt, "UNet")
        self.discriminator = models.define_discriminator(opt, "PatchGan")
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, loss_weights=[0.5])
        #self.gan = models.define_gan(self.generator, self.discriminator, self.opt)
        
        # make weights in the discriminator not trainable
        for layer in self.discriminator.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False

        in_src = Input(shape=opt.input_shape) # Define the source image
        mask_src = Input(shape = opt.output_shape) # Adding mask as an input 
        gen_out = self.generator(in_src) # Connect the source image to the generator input
        
        if not opt.no_bim:
            #print("Warning: bim is on in custom_model.")
            gen_out = tf.math.multiply(gen_out - 1, mask_src) + 1
            tf.cast(gen_out, dtype = tf.float32)
        
        # connect the source input and generator output to the discriminator input
        dis_out = self.discriminator([in_src, gen_out])

        outs = [dis_out, gen_out]
        loss=['binary_crossentropy', 'mae']
        loss_weights=[0.01, 1]

        optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.gan = Model([in_src, mask_src], outs)
        self.gan.compile(loss=loss, optimizer=optimizer, loss_weights=loss_weights)


    def train_on_batch(self, opt, X_realA, X_realB, X_fakeB, mask_batch, y_real, y_fake):
        # Select Targets that Match Model Outputs
        outs = models.retrieve_outputs(opt, X_realB, mask_batch, y_real)
        # Train Models
        _ = self.discriminator.train_on_batch([X_realA, X_realB], y_real)
        _ = self.discriminator.train_on_batch([X_realA, X_fakeB], y_fake)
        _ = self.gan.train_on_batch([X_realA, mask_batch], outs)
