# Matthieu Dagommer
# 04/07/2022

# Custom Class to define our own architecture

from models.base_model import BaseModel
from models.networks import gram_matrix, mpd_loss, mpdLoss
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
import keras.backend as K

class Custom(BaseModel):

    def __init__(self, opt):

        super(Custom, self).__init__(opt)
        optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.generator = models.define_generator(opt, opt.netG)
        self.discriminator = models.define_discriminator(opt, opt.netD)

        if opt.discls:
            disc_loss = 'mse'
            self.discriminator.compile(loss='mse', optimizer=optimizer, loss_weights=[0.5])
        else:
            disc_loss = 'binary_crossentropy'
            self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, loss_weights=[0.5])

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
        if opt.perceptual_loss:
            for layer in self.discriminator.layers:
                if not isinstance(layer, BatchNormalization):
                    layer.trainable = False

            if opt.discls:
                dis_outs = self.discriminator(gen_out)
            else:
                dis_outs = self.discriminator([in_src, gen_out])
            
            dis_out = dis_outs[0]
        else:
            if opt.discls:
                dis_out = self.discriminator(gen_out)
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
    
        # Generator losses 
        
        loss_metric = opt.loss_metric

        if opt.gdl:
            # Compute Gradients
            gen_dx, gen_dy, gen_dz = utils.tf_gradient_3d(gen_out)

            if opt.l1_l2:
                outs = [dis_out, gen_out, gen_out, gen_dx, gen_dy, gen_dz]
                loss = [disc_loss, 'mae', 'mse', opt.gdl_metric, opt.gdl_metric, opt.gdl_metric]
                loss_weights = [0.01, 1, 1, 1, 1, 1]
            else:
                outs = [dis_out, gen_out, gen_dx, gen_dy, gen_dz]
                loss = [disc_loss, loss_metric, opt.gdl_metric, opt.gdl_metric, opt.gdl_metric]
                loss_weights = [0.01, 1, 1, 1, 1]
        else:
            if opt.l1_l2:
                outs = [dis_out, gen_out, gen_out]
                loss=[disc_loss, 'mae', 'mse']
                loss_weights=[0.01, 1, 1]
            else:
                outs = [dis_out, gen_out]
                loss=[disc_loss, loss_metric]
                loss_weights=[0.01, 1]
        
        # Feature Losses
        if opt.feature_loss:

            # Content Loss
            content_losses = ['mse' for i in range(len(gen_out_features))]
            loss += content_losses
            content_loss_weights = [1 / len(gen_out_features) / 2 for i in range(len(gen_out_features))]
            loss_weights += content_loss_weights
            outs += gen_out_features

            """
            # Style Loss
            style_losses = ['mse' for i in range(len(gen_out_features_gram))]
            loss += style_losses
            style_loss_weights = [1 / len(gen_out_features) / 2 for i in range(len(gen_out_features))]
            loss_weights += style_loss_weights
            outs += gen_out_features_gram
            """
            """
            print("feature loss weights:")
            print("content: ", content_loss_weights)
            print("style: ", style_loss_weights)
            """

        # Perceptual Loss
        if opt.perceptual_loss:
            perceptual_losses = ['mae' for i in range(len(dis_outs)-1)]
            loss += perceptual_losses
            perceptual_loss_weights = [1/(len(dis_outs)-1) for i in range(len(dis_outs)-1)]
            loss_weights += perceptual_loss_weights
            outs += dis_outs[1:] # Remove the output patch of the discriminator
        
        loss_weights = np.array(loss_weights)
        loss_weights[1:] = 1 / np.sum(loss_weights[1:]) # keep a 1/100 ratio b/w disc loss and gen loss
        loss_weights = loss_weights.tolist()

        if opt.loss_metric == 'mpd':
            gen_dx, gen_dy, gen_dz = utils.tf_gradient_3d(gen_out)
            outs = [dis_out, gen_out, gen_dx, gen_dy, gen_dz]
            gdl_coef = 10 / 3
            loss = [disc_loss, mpdLoss(), opt.gdl_metric, opt.gdl_metric, opt.gdl_metric]
            loss_weights_raw = [1, 30, gdl_coef, gdl_coef, gdl_coef]
            loss_weights = [i / sum(loss_weights_raw) for i in loss_weights_raw]
        
        self.gan = Model([in_src, mask_src], outs)
        self.gan.compile(loss=loss, optimizer=optimizer, loss_weights=loss_weights)


    def train_on_batch(self, opt, X_realA, X_realB, X_fakeB, mask_batch, y_real, y_fake):
        # Select Targets that Match Model Outputs
        outs = models.retrieve_outputs(opt, X_realB, mask_batch, y_real, self.vgg16, X_realA)
        # Add perceptual loss
        if opt.perceptual_loss:
            if opt.discls:
                dis_outs = self.discriminator(X_realB)
            else:
                dis_outs = self.discriminator([X_realA, X_realB])
            percep_layers = dis_outs[1:]
            outs += percep_layers
        # Train Models
        if opt.discls:
            _ = self.discriminator.train_on_batch(X_realB, y_real)
            _ = self.discriminator.train_on_batch(X_fakeB, y_fake)
        else:
            _ = self.discriminator.train_on_batch([X_realA, X_realB], y_real)
            _ = self.discriminator.train_on_batch([X_realA, X_fakeB], y_fake)

        _ = self.gan.train_on_batch([X_realA, mask_batch], outs)

"""
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
"""