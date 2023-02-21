# Matthieu Dagommer
# 04/07/2022

# Custom Class to define our own architecture

from models.base_model import BaseModel
from models.networks import gram_matrix
import utils.utils as utils
import models.networks as models
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras import Input
from keras.layers import Conv2D
from keras.layers import BatchNormalization
import numpy as np
import time

# Fila-sGAN specifics:
# - it has content/style losses w/ a VGG network
# - generator/discriminator are slightly different than usual UNet and PatchGAN  


class FilaSGan(BaseModel):

    def __init__(self, opt):
        
        # FilaSGan specifics:

        opt.netG = "FilaSGan"
        opt.netD = "FilaSGan"
        opt.perceptual_loss = False
        opt.feature_loss = True
        opt.gdl = False
        opt.l1_l2 = False


        super(FilaSGan, self).__init__(opt)
        optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.generator = models.define_generator(opt, opt.netG)
        self.discriminator = models.define_discriminator(opt, opt.netD)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, loss_weights=[0.5])

        # make weights in the discriminator not trainable
        for layer in self.discriminator.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False

        in_src = Input(shape=opt.input_shape) # Define the source image
        mask_src = Input(shape = opt.output_shape) # Adding mask as an input 
        gen_out = self.generator(in_src) # Connect the source image to the generator input
        
        # ------- ADDING A MASK STEP --------- #
        gen_out = tf.math.multiply(gen_out - 1, mask_src) + 1
        tf.cast(gen_out, dtype = tf.float32)
        
        # connect the source input and generator output to the discriminator input
        if opt.perceptual_loss:
            dis_outs = self.discriminator([in_src, gen_out])
            dis_out = dis_outs[0]
        else:
            dis_out = self.discriminator([in_src, gen_out])
        
        # Feature Loss (Content + Style Loss)
        if opt.feature_loss:
            self.vgg16 = models.define_VGG16(opt)

            for layer in self.vgg16.layers:
                layer.trainable = False # Make weights in the vgg16 not trainable
            
            # Retrieve feature maps from pretrained VGG16
            gen_out_rgb = models.vgg16_preprocess(gen_out[:,:,:,0])
            gen_out_features = self.vgg16(gen_out_rgb)

            # Compute Gram Matrices
            gen_out_features_gram = []
            for i in range(len(gen_out_features)):
                gen_out_features_gram.append(gram_matrix(gen_out_features[i]))
            
        else:
            self.vgg16 = None

        optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        
        if opt.gdl:
            # Compute Gradients
            gen_dx, gen_dy, gen_dz = utils.tf_gradient_3d(gen_out)

            if opt.l1_l2:
                outs = [dis_out, gen_out, gen_out, gen_dx, gen_dy, gen_dz]
                loss = ['binary_crossentropy', 'mae', 'mse', opt.gdl_metric, opt.gdl_metric, opt.gdl_metric]
                loss_weights = [0.01, 1, 1, 1, 1]
            else:
                outs = [dis_out, gen_out, gen_dx, gen_dy, gen_dz]
                loss = ['binary_crossentropy', opt.loss_metric, opt.gdl_metric, opt.gdl_metric, opt.gdl_metric]
                loss_weights = [0.01, 1, 1, 1]
        else:
            if opt.l1_l2:
                outs = [dis_out, gen_out, gen_out]
                loss=['binary_crossentropy', 'mae', 'mse']
                loss_weights=[0.01, 1, 1]
            else:
                outs = [dis_out, gen_out]
                loss=['binary_crossentropy',opt.loss_metric]
                loss_weights=[0.01, 1]
        
        # Feature Losses
        if opt.feature_loss:

            # Content Loss
            content_losses = ['mse' for i in range(len(gen_out_features))]
            loss += content_losses
            content_loss_weights = [1 / len(gen_out_features) for i in range(len(gen_out_features))]
            loss_weights += content_loss_weights
            outs += gen_out_features

            # Style Loss
            style_losses = ['mse' for i in range(len(gen_out_features_gram))]
            loss += style_losses
            style_loss_weights = [1 / len(gen_out_features_gram) for i in range(len(gen_out_features_gram))]
            loss_weights += style_loss_weights
            outs += gen_out_features_gram
        
        # Perceptual Loss
        if opt.perceptual_loss:
            perceptual_losses = ['mae' for i in range(len(dis_outs)-1)]
            loss += perceptual_losses
            perceptual_loss_weights = [1/(len(dis_outs)-1) for i in range(len(dis_outs)-1)]
            loss_weights += perceptual_loss_weights
            outs += dis_outs[1:] # Remove the output patch of the discriminator

        self.gan = Model([in_src, mask_src], outs)
        self.gan.compile(loss=loss, optimizer=optimizer, loss_weights=loss_weights)


    def generate_fake_samples(self, input, disc_shape, acm_model = None):

        output = self.batch_computation(input)
        disc_y_shape = tuple([len(output)] + list(disc_shape))
        y_fake = np.zeros(shape=disc_y_shape)
        return output, y_fake


    def train_on_batch(self, opt, X_realA, X_realB, X_fakeB, mask_batch, y_real, y_fake):
        # Select Targets that Match Model Outputs
        outs = models.retrieve_outputs(opt, X_realB, mask_batch, y_real, self.vgg16, X_realA)
        # Add perceptual loss
        if opt.perceptual_loss:
            dis_outs = self.discriminator([X_realA, X_realB])
            percep_layers = dis_outs[1:]
            outs += percep_layers
        # Train Models
        _ = self.discriminator.train_on_batch([X_realA, X_realB], y_real)
        _ = self.discriminator.train_on_batch([X_realA, X_fakeB], y_fake)
        _ = self.gan.train_on_batch([X_realA, mask_batch], outs)
        
        
    def compute_train_error(self, opt, real, fake, mask):
        if opt.sqrt_poro:
            diff = np.square(fake) - np.square(real)
        else:
            diff = fake - real
        errors_flat = diff.flatten()
        mask_flat = mask.flatten()
        n_mask_pixels = int(mask_flat.sum())
        train_error = np.abs(errors_flat[np.where(mask_flat == 1)]).sum()
        return train_error, n_mask_pixels

    def compute_valid_error(self, opt, source, target, mask):

        generated = self.batch_computation(source)
        if opt.sqrt_poro:
            valid_errors_matrix = np.square(generated) - np.square(target)
        else:
            valid_errors_matrix = generated - target
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