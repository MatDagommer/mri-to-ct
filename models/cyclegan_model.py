# Matthieu Dagommer
# 04/07/2022

# Custom Class to define our own architecture

from models.base_model import BaseModel
import models.networks as models
import utils.utils as utils
from tensorflow import keras
import numpy as np
from keras.models import Model
from keras import Input

class CycleGan(BaseModel):

    def __init__(self, opt):
        
        super(CycleGan, self).__init__(opt)

        # Input shape
        self.img_shape = opt.input_shape

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss

        optimizer = keras.optimizers.Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.discriminator = models.define_discriminator(opt, "CycleGan", self.img_shape, self.df)
        self.d_B = models.define_discriminator(opt, "CycleGan", self.img_shape, self.df)
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.generator = models.define_generator(opt, "ResNet9")
        self.g_BA = models.define_generator(opt, "ResNet9")

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.generator(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.generator(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.generator(img_B)

        # For the gan model we will only train the generators
        self.discriminator.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.discriminator(fake_A)
        valid_B = self.d_B(fake_B)

        # gan model trains generators to fool discriminators
        self.gan = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])

        self.gan.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)


    def train_on_batch(self, opt, X_realA, X_realB, X_fakeB, mask_batch, y_real, y_fake):

        # Select Targets that Match Model Outputs
        #outs = models.retrieve_outputs(opt, X_realB, mask_batch, y_real)
        
        # ----------------------
        #  Train Discriminators
        # ----------------------

        # Translate images to opposite domain
        fake_B = self.generator.predict(X_realA)#, y_real)
        fake_A = self.g_BA.predict(X_fakeB)#, y_fake)

        # Train the discriminators (original images = real / translated = Fake)
        _ = self.discriminator.train_on_batch(X_realA, y_real)
        _ = self.discriminator.train_on_batch(fake_A, y_fake)

        _ = self.d_B.train_on_batch(X_realB, y_real)
        _ = self.d_B.train_on_batch(fake_B, y_fake)

        # ------------------
        #  Train Generators
        # ------------------
        # Train the generators

        _ = self.gan.train_on_batch([X_realA, X_realB],
                                                [y_real, y_real,
                                                X_realA, X_realB,
                                                X_realA, X_realB])

    #def generate_fake_samples(self, input, patch_shape):
    #    output = self.generator.predict(input, verbose = 0)
    #    y_fake = np.zeros((len(output), patch_shape, patch_shape, 1))
    #    return output, y_fake
"""
    def generate_fake_samples(self, input, disc_shape):# acm_model = None):

        output = self.batch_computation(input)
        disc_y_shape = tuple([len(output)] + list(disc_shape))
        y_fake = np.zeros(shape=disc_y_shape)
        return output, y_fake


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
"""
    def compute_valid_error(self, opt, source, target, mask_flat, n_mask_pixels):
        generated = self.generator.predict(source, verbose = 0)    
        if opt.sqrt_poro:
            valid_errors_matrix = np.square(generated) - np.square(target)
        else:
            valid_errors_matrix = generated - target
        valid_errors_flat = valid_errors_matrix.flatten()
        val_error = (1/(n_mask_pixels))*np.abs(valid_errors_flat[np.where(mask_flat == 1)]).sum()
        return generated, valid_errors_flat, val_error


    def compute_train_error(self, opt, trainA, trainB, M_train, n_patch):
        # generate random subset of samples with size <opt.ns_valid>
        sources, targets, coord, _ = utils.generate_real_samples(trainA, trainB, opt.ns_valid, \
                n_patch, opt.n_subjects, opt.to, opt.flip, opt.set_pCT, opt.patch25d)
        # Generate predictions ("fakes") with the model
        generated, _ = self.generate_fake_samples(sources, n_patch)
        # Retrieve Masks that correspond with the inputs
        mask_batch = models.retrieve_masks(opt, coord, M_train, sources.shape)
        mask_flat = mask_batch.flatten()
        n_train_mask_pixels = int(mask_flat.sum())
        # compute total error
        if opt.sqrt_poro:
            train_errors_matrix = np.square(generated) - np.square(targets)
        else:
            train_errors_matrix = generated - targets
        train_errors_flat = train_errors_matrix.flatten()
        train_error = (1/(n_train_mask_pixels))*np.abs(train_errors_flat[np.where(mask_flat == 1)]).sum()
        return train_error


    def save_model(self, opt):
        self.generator.save(opt.model_path + opt.name + "_g.h5")
        #self.gan.save(opt.model_path + opt.name + "_gan.h5")
        #self.discriminator.save(opt.model_path + opt.name + "_d.h5")

    def get_summary(self, opt):
        with open(opt.model_path + 'gan_summary.txt','w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.gan.summary(print_fn=lambda x: fh.write(x + '\n'))

    def generate_images(self, test_set, mm_ct):
        utils.generate_images(self.generator, test_set, mm_ct)
    
    def discriminator_output_shape(self):
        return self.discriminator.output_shape[-1]

"""