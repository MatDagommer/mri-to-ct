# Matthieu Dagommer
# 27/06/2022

# Model utilities

import numpy as np
#from models.filasgan_model import FilaSGan
from utils import utils
import tensorflow as tf
from tensorflow import keras
from keras.initializers import RandomNormal
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
from keras.activations import sigmoid
import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from tensorflow.python.ops import math_ops
import pickle



class ReflectionPadding2D(Layer):
    
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'padding': self.padding
        })
        return config



def ResNet9(opt, use_dropout=False, padding_type='reflect'):
	
	output_nc = opt.output_shape[-1]
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=opt.input_shape)
	padding = "valid"
    
	# ReflectionPad2d(3)
	p1 = ReflectionPadding2D((3,3))(in_image)
	c1 = Conv2D(64, kernel_size = (7,7), strides = (1,1), padding = padding)(p1)
	b1 = BatchNormalization()(c1, training=True) # Training: normalizes wrt current batch input
	a1 = Activation('relu')(b1)
    
	p2 = ZeroPadding2D(padding=(1, 1))(a1)
	c2 = Conv2D(128, kernel_size = (3,3), strides = (2,2), padding = padding)(p2) # padding of 1 in Pytorch ???
	b2 = BatchNormalization()(c2, training=True)
	a2 = Activation('relu')(b2)
    
	p3 = ZeroPadding2D(padding=(1, 1))(a2)
	c3 = Conv2D(256, kernel_size = (3,3), strides = (2,2), padding = padding)(p3)
	b3 = BatchNormalization()(c3, training=True)
	a3 = Activation('relu')(b3)
    
	r1 = a3 + resnet_block(a3, 256, batchnorm = True, dropout = False)
	r2 = r1 + resnet_block(r1, 256, batchnorm = True, dropout = False)
	r3 = r2 + resnet_block(r2, 256, batchnorm = True, dropout = False)
	r4 = r3 + resnet_block(r3, 256, batchnorm = True, dropout = False)
	r5 = r4 + resnet_block(r4, 256, batchnorm = True, dropout = False)
	r6 = r5 + resnet_block(r5, 256, batchnorm = True, dropout = False)
	r7 = r6 + resnet_block(r6, 256, batchnorm = True, dropout = False)
	r8 = r7 + resnet_block(r7, 256, batchnorm = True, dropout = False)
	r9 = r8 + resnet_block(r8, 256, batchnorm = True, dropout = False)
    
	#x = ZeroPadding2D(padding=(1, 1))(r9)
	x = Conv2DTranspose(128, kernel_size = (3,3), strides = (2,2), output_padding = 1)(r9)
	x = BatchNormalization()(x, training=True) # Training: normalizes wrt current batch input
	x = Activation('relu')(x)
	x = Cropping2D((1,1))(x)
    
	#x = ZeroPadding2D(padding=(1, 1))(x)
	x = Conv2DTranspose(64, kernel_size = (3,3), strides = (2,2), output_padding = 1)(x) # padding of 1 in Pytorch ???
	x = BatchNormalization()(x, training=True)
	x = Activation('relu')(x)
	x = Cropping2D((1,1))(x)
    
	x = ReflectionPadding2D((3,3))(x)
	x = Conv2D(output_nc, kernel_size = (7,7), strides = (1,1), padding = padding)(x)
	#out_image = Activation('sigmoid')(x) # original ResNet has tanh
	if opt.tanh:
		out_image = Activation('tanh')(g)
	else:
		out_image = Activation('sigmoid')(g)
	# define model
	model = Model(in_image, out_image)
	return model



def filter_x(shape, dtype = None):
    f = np.array([-1, 1])
    f = f.reshape(1,2,1,1)
    #assert f.shape == shape
    return K.variable(f, dtype='float32')



def filter_y(shape, dtype =None):
    f = np.array([[1], [-1]])
    f = f.reshape(2,1,1,1)
    #assert f.shape == shape
    return K.variable(f, dtype='float32')



def resnet_block(layer_in, dim, batchnorm = True, dropout = False):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    padding = "valid"
    
    x = ReflectionPadding2D((1,1))(layer_in)
    x = Conv2D(dim, kernel_size = (3,3), strides = (1,1), padding = padding)(x)
    x = BatchNormalization()(x, training = True)
    x = Activation('relu')(x)
    if dropout:
        x = Dropout(0.5)(x, training=True)
    x = ReflectionPadding2D((1,1))(x)
    x = Conv2D(dim, kernel_size = (3,3), strides = (1,1), padding = padding)(x)
    x = BatchNormalization()(x, training = True)
    return x


# define the discriminator model
def Discriminator(opt):#image_shape, output_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=opt.input_shape)
	# target image input
	in_target_image = Input(shape=opt.output_shape)
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64 
	d1 = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    # 64 filters, kernel_size = (4,4), strides, padding = 'same' means the output size is same as input size
    # means that there is even zero-padding left right top bottom.
	d2 = LeakyReLU(alpha=0.2)(d1)
	# C128
	d3 = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d2)
	d4 = BatchNormalization()(d3)
	d5 = LeakyReLU(alpha=0.2)(d4)
	# C256
	d6 = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d5)
	d7 = BatchNormalization()(d6)
	d8 = LeakyReLU(alpha=0.2)(d7)
	# C512
	d9 = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d8)
	d10 = BatchNormalization()(d9)
	d11 = LeakyReLU(alpha=0.2)(d10)
	# second last output layer
	d12 = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d11)
	d13 = BatchNormalization()(d12)
	d14 = LeakyReLU(alpha=0.2)(d13)
	# patch output
	d15 = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d14)
	patch_out = Activation('sigmoid')(d15)
	# define model
	if opt.perceptual_loss:
		model = Model([in_src_image, in_target_image], [patch_out, d2, d5, d8, d11])
	else:
		model = Model([in_src_image, in_target_image], patch_out)
	return model

# define the discriminator model
def Discriminator_Koh(opt):#image_shape, output_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=opt.input_shape)
	# C64 
	d1 = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_src_image)
    # 64 filters, kernel_size = (4,4), strides, padding = 'same' means the output size is same as input size
    # means that there is even zero-padding left right top bottom.
	d2 = LeakyReLU(alpha=0.2)(d1)
	# C128
	d3 = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d2)
	#d4 = BatchNormalization()(d3)
	d4 = InstanceNormalization()(d3)
	d5 = LeakyReLU(alpha=0.2)(d4)
	# C256
	d6 = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d5)
	#d7 = BatchNormalization()(d6)
	d7 = InstanceNormalization()(d6)
	d8 = LeakyReLU(alpha=0.2)(d7)
	# C512
	d9 = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d8)
	#d10 = BatchNormalization()(d9)
	d10 = InstanceNormalization()(d9)
	d11 = LeakyReLU(alpha=0.2)(d10)
	# second last output layer
	d12 = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d11)
	#d13 = BatchNormalization()(d12)
	d13 = InstanceNormalization()(d12)
	d14 = LeakyReLU(alpha=0.2)(d13)
	# patch output
	d15 = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d14)
	patch_out = Activation('sigmoid')(d15)
	# define model
	if opt.perceptual_loss:
		model = Model(in_src_image, [patch_out, d2, d5, d8, d11])
	else:
		model = Model(in_src_image, patch_out)
	return model

# CycleGAN Discriminator
def CycleGanDiscriminator(input_shape, n_filters):

	def d_layer(layer_input, filters, f_size=4, normalization=True):
		"""Discriminator layer"""
		d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
		d = LeakyReLU(alpha=0.2)(d)
		if normalization:
			d = InstanceNormalization()(d)
		return d

	img = Input(shape=input_shape)

	d1 = d_layer(img, n_filters, normalization=False)
	d2 = d_layer(d1, n_filters*2)
	d3 = d_layer(d2, n_filters*4)
	d4 = d_layer(d3, n_filters*8)
	validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
	return Model(img, validity)

# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g

# define the standalone generator model
def UNet(opt):#input_shape, output_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=opt.input_shape)
	# encoder model
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	# decoder model
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# output
	g = Conv2DTranspose(opt.output_shape[-1], (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	if opt.tanh:
		out_image = Activation('tanh')(g)
	else:
		out_image = Activation('sigmoid')(g)
	# define model
	model = Model(in_image, out_image)
	return model

# define the standalone generator model
def UNet128(opt):#input_shape, output_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=opt.input_shape)
	# encoder model
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e6)
	b = Activation('relu')(b)
	# decoder model
	d1 = decoder_block(b, e6, 512)
	d2 = decoder_block(d1, e5, 512)
	d3 = decoder_block(d2, e4, 512)
	d4 = decoder_block(d3, e3, 256, dropout=False)
	d5 = decoder_block(d4, e2, 128, dropout=False)
	d6 = decoder_block(d5, e1, 64, dropout=False)
	# output
	g = Conv2DTranspose(opt.output_shape[-1], (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d6)
	#out_image = Activation('sigmoid')(g)
	if opt.tanh:
		out_image = Activation('tanh')(g)
	else:
		out_image = Activation('sigmoid')(g)
	# define model
	model = Model(in_image, out_image)
	return model

# define the standalone generator model
def UNet64(opt):#input_shape, output_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=opt.input_shape)
	# encoder model
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e5)
	b = Activation('relu')(b)
	# decoder model
	d1 = decoder_block(b, e5, 512)
	d2 = decoder_block(d1, e4, 512)
	d3 = decoder_block(d2, e3, 256)
	d4 = decoder_block(d3, e2, 128, dropout=False)
	d5 = decoder_block(d4, e1, 64, dropout=False)
	# output
	g = Conv2DTranspose(opt.output_shape[-1], (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d5)
	#out_image = Activation('sigmoid')(g)
	if opt.tanh:
		out_image = Activation('tanh')(g)
	else:
		out_image = Activation('sigmoid')(g)
	# define model
	model = Model(in_image, out_image)
	return model

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples, verbose = 0)
    # create 'fake' class labels (0)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


def define_generator(opt, generator):
    if generator == "UNet":
        netG = UNet(opt)
    elif generator == "ResNet9":
        netG = ResNet9(opt)
    elif generator == "FilaSGan":
        netG = FilaSGanGenerator(opt)
    elif generator == "UNet64":
        netG = UNet64(opt)
    elif generator == "UNet128":
        netG = UNet128(opt)
    elif generator == "ResNet9_3D":
        netG = ResNet9_3D(opt)
    return netG


def define_discriminator(opt, discriminator, input_shape = None, n_filters = None):
	if discriminator == "PatchGan":
		netD = Discriminator(opt)
	if discriminator == "Discriminator_Koh":
		netD = Discriminator_Koh(opt)
	elif discriminator == "CycleGan":
		netD = CycleGanDiscriminator(input_shape, n_filters)
	elif discriminator == "FilaSGan":
		netD = FilaSGanDiscriminator(opt)
	elif discriminator == "Discriminator_3D":
		netD = Discriminator_3D(opt)
	elif discriminator == 'Discriminator_3D_Koh':
		netD = Discriminator_3D_Koh(opt)
	return netD


def valid_error(g_model, real, target, mask_flat, n_mask_pixels, opt):

    generated = g_model.predict(real, verbose = 0)    
    if not opt.no_sqrt:
        valid_errors_matrix = np.square(generated) - np.square(target)
    else:
        valid_errors_matrix = generated - target   

    valid_errors_flat = valid_errors_matrix.flatten()
    val_error = (1/(n_mask_pixels))*np.abs(valid_errors_flat[np.where(mask_flat == 1)]).sum()
    return generated, valid_errors_flat, val_error


def retrieve_outputs(opt, targets, masks, y_real, vgg16 = None, sources = None):
	
    if opt.gdl:
        targets = tf.cast(targets, dtype = tf.float32)
        #if opt.masked_input:
        gt_dx, gt_dy, gt_dz = utils.tf_gradient_3d(targets)
        #else:
        #    targets_masked = np.multiply(targets - 1, masks) + 1
        #    gt_dx, gt_dy, gt_dz = utils.tf_gradient_3d(targets_masked)
        if opt.l1_l2:
            outs = [y_real, targets, targets, gt_dx, gt_dy, gt_dz] 
        else:
            outs = [y_real, targets, gt_dx, gt_dy, gt_dz]
    else:
        if opt.l1_l2:
            outs = [y_real, targets, targets]
        else:
            outs = [y_real, targets]
	
    if opt.feature_loss:
		# Content Loss
        sources_rgb = vgg16_preprocess(sources[:,:,:,0])
        source_features = vgg16.predict(sources_rgb)
        outs += source_features

		# Style Loss
        source_features_gram = []
        for i in range(len(source_features)):
            source_features_gram.append(gram_matrix(source_features[i]))
        outs += source_features_gram

	#if opt.perceptual_loss:
    if opt.loss_metric == 'mpd':
        targets = tf.cast(targets, dtype = tf.float32)
        gt_dx, gt_dy, gt_dz = utils.tf_gradient_3d(targets)
        outs = [y_real, targets, gt_dx, gt_dy, gt_dz]
	
    return outs

def define_VGG16(opt):

	#in_image = Input(shape=opt.input_shape)
	vgg16_model = VGG16(weights = 'imagenet', include_top = False, input_shape = (256,256,3))

	# Indices of main convolutional layers
	ids = [2, 5, 9, 13, 17]
	outputs = [vgg16_model.layers[i].output for i in ids]
	model = Model(inputs = vgg16_model.inputs, outputs = outputs)
	return model

def vgg16_preprocess(input):
	# Get CT in RGB format [0; 255]
	# 0 is 0 and 1 is 255
	#input_norm = input * 255
	input_rgb = tf.stack((input,)*3, axis = -1)
	# Actual Centering makes little sense with the current data distribution
	# (not a gaussian curve at all, high concentration at poro ~ 1)
	input_rgb_centered = (input_rgb - 1) / 2 
	return input_rgb_centered


def gram_matrix(input_tensor):
	result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
	input_shape = tf.shape(input_tensor)
	num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
	return result/(num_locations)

def FilaSGanGenerator(opt):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=opt.input_shape)
	# random input
	z = Lambda(lambda x: K.random_normal((None, 400), stddev=0.02, dtype = tf.float32))

	# 256
	e1 = Conv2D(64, (4,4), strides = (2,2), padding = "same", kernel_initializer=init)(in_image)
	a1 = LeakyReLU(alpha=0.2)(e1)
	# 128
	e2 = Conv2D(128, (4,4), strides = (2,2), padding = "same", kernel_initializer=init)(a1)
	b1 = BatchNormalization()(e2, training=True)
	a2 = LeakyReLU(alpha=0.2)(b1)
	# 64
	e3 = Conv2D(256, (4,4), strides = (2,2), padding = "same", kernel_initializer=init)(a2)
	b2 = BatchNormalization()(e3, training=True)
	a3 = LeakyReLU(alpha=0.2)(b2)
	# 32
	e4 = Conv2D(512, (4,4), strides = (2,2), padding = "same", kernel_initializer=init)(a3)
	b3 = BatchNormalization()(e4, training=True)
	a4 = LeakyReLU(alpha=0.2)(b3)
	# 16
	e5 = Conv2D(512, (4,4), strides = (2,2), padding = "same", kernel_initializer=init)(a4)
	b4 = BatchNormalization()(e5, training=True)
	a5 = LeakyReLU(alpha=0.2)(b4)
	# 8

	# Add random noise
	z2 = Dense(4*4*64, kernel_initializer=init)(z)
	a6 = LeakyReLU(alpha=0.2)(z2)
	z3 = tf.reshape(a6, [-1, 4, 4, 64])
	a7 = LeakyReLU(alpha=0.2)(z3)
	g = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(a7)
	g = BatchNormalization()(g, training=True)
	g = Concatenate()([g, a5])
	g = LeakyReLU(alpha=0.2)(g)
	# 8
	g = Conv2DTranspose(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = BatchNormalization()(g, training=True)
	g = Concatenate()([g, a4])
	g = LeakyReLU(alpha=0.2)(g)
	# 16
	d1 = decoder_block(g, a3, 512, dropout=False)
	# 32
	d2 = decoder_block(d1, a2, 256, dropout=False)
	# 64
	d3 = decoder_block(d2, a1, 128, dropout=False)
	# 128
	d4 = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d3)
	#out_image = Activation('sigmoid')(d4)
	if opt.tanh:
		out_image = Activation('tanh')(d4)
	else:
		out_image = Activation('sigmoid')(d4)
	# 256

	model = Model([in_image, z], out_image)
	return model


def FilaSGanDiscriminator(opt):

	in_src_image = Input(opt.input_shape)
	in_target_image = Input(opt.output_shape)
	init = RandomNormal(stddev=0.02)
	# 256
	d1 = Conv2D(32, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_src_image)
	d2 = LeakyReLU(alpha=0.2)(d1)
	# 128
	d3 = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d2)
	d4 = BatchNormalization()(d3)
	d5 = LeakyReLU(alpha=0.2)(d4)
	# 64
	d6 = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d5)
	d7 = BatchNormalization()(d6)
	d8 = LeakyReLU(alpha=0.2)(d7)
	# 32
	d9 = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d8)
	d10 = BatchNormalization()(d9)
	d11 = LeakyReLU(alpha=0.2)(d10)
	# 16
	d12 = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d11)
	d13 = BatchNormalization()(d12)
	d14 = LeakyReLU(alpha=0.2)(d13)
	d15 = Flatten()(d14)
	# Fully-Connected
	d16 = Dense(1, kernel_initializer=init)(d15)
	patch_out = Activation('sigmoid')(d16)
	model = Model([in_src_image, in_target_image], patch_out)
	return model



# -------------------- ID-CGAN -------------------- #
# Retrieved from https://github.com/wazeerzulfikar/IDC-GAN/blob/master/model.py

import functools
from functools import partial
from functools import reduce

_Conv2D = partial(Conv2D, padding="same")
_Conv2DTranspose = partial(Conv2DTranspose, padding="same")


def compose(*funcs):
    if funcs:
    	return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)

def LR_Conv_BN(*args, **kwargs):
	return compose(
		LeakyReLU(0.02),
		_Conv2D(*args,**kwargs),
		BatchNormalization()
		)	


def R_DeConv_BN(*args, **kwargs):
	return compose(
		Activation('relu'),
		_Conv2DTranspose(*args, **kwargs),
		BatchNormalization()
		)


def create_idcgan_generator(input_nc, output_nc, ngf):

	inputs = Input(shape=(256,256,1))

	#Encoder

	e1 = _Conv2D(ngf, (3,3), strides=1)(inputs)

	e2 = LR_Conv_BN(ngf, (3,3), strides=1)(e1)

	e3 = LR_Conv_BN(ngf, (3,3), strides=1)(e2)

	e4 = LR_Conv_BN(ngf, (3,3), strides=1)(e3)

	e5 = LR_Conv_BN(int(ngf/2), (3,3), strides=1)(e4)

	e6 = LR_Conv_BN(1, (3,3), strides=1)(e5)

	#Decoder

	d1 = LeakyReLU(0.02)(e6)
	d1 = _Conv2DTranspose(int(ngf/2), (3,3), strides=1)(d1)
	d1 = BatchNormalization()(d1)

	d2 = R_DeConv_BN(ngf, (3,3), strides=1)(d1)
	d2 = Add()([d2,e4])

	d3 = R_DeConv_BN(ngf, (3,3), strides=1)(d2)

	d4 = R_DeConv_BN(ngf, (3,3), strides=1)(d3)
	d4 = Add()([d4,e2])

	d5 = R_DeConv_BN(ngf, (3,3), strides=1)(d4)

	d6 = Activation('relu')(d5)
	d6 = _Conv2DTranspose(output_nc, (3,3), strides=1)(d6)

	o1 = Activation('tanh')(d6)

	model = Model(inputs,o1)

	return model


def create_idcgan_discriminator(input_nc, output_nc, ndf, n_layers):

	model = Sequential()

	model.add(_Conv2D(ndf, (4,4), strides=2, input_shape=(256,256,input_nc+output_nc)))
	model.add(LeakyReLU(0.2))

	for i in range(1,n_layers-1):
		nf_mult = min(2**i,8)
		model.add(_Conv2D(ndf*nf_mult,(4,4),strides=2))
		model.add(BatchNormalization())
		model.add(LeakyReLU(0.2))

	nf_mult = min(2**n_layers, 8)
	model.add(_Conv2D(ndf*nf_mult, (4,4), strides=2))
	model.add(BatchNormalization())
	model.add(_Conv2D(1, (4,4), strides=1))
	model.add(Activation('sigmoid'))

	return model

	
def generator_containing_discriminator(generator, discriminator):
    inputs = Input(shape=(256,256,3))
    x_generator = generator(inputs)
    
    merged = Concatenate(axis=3)([inputs, x_generator])
    discriminator.trainable = False
    x_discriminator = discriminator(merged)

    concatenated = Concatenate(axis=0)([inputs, x_generator])
    base_model = VGG16(weights='imagenet', include_top=False)

    vgg_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_conv2').output)
    vgg_model.trainable = False

    vgg_model_out = vgg_model(concatenated)
    
    model = Model(inputs=inputs, outputs=[x_generator,x_discriminator,vgg_model_out])
    
    return model, x_generator, x_discriminator, vgg_model_out


lambda1 = 6.6e-3
lambda2 = 1

def idcgan_constant_loss(y_true,y_pred):
	return K.constant(1)

def idcgan_perceptual_loss(vgg_out):
	actual_features, generated_features = tf.split(vgg_out,2)
	loss = K.mean(K.square(K.flatten(generated_features) - K.flatten(actual_features)), axis=-1)
	return loss

def idcgan_entropy_loss(d_out):
	return -1*K.mean(K.log(K.flatten(d_out)))

def idcgan_discriminator_loss(y_true,y_pred):
    return K.mean(K.binary_crossentropy(K.flatten(y_pred), K.flatten(y_true)), axis=-1)

def idcgan_generator_l2_loss(y_true,y_pred):
    return K.mean(K.square(K.flatten(y_pred) - K.flatten(y_true)), axis=-1)


def idcgan_refined_perceptual_loss(y_true, y_pred):
	return idcgan_discriminator_loss(y_true, y_pred)+lambda1*idcgan_perceptual_loss(y_true,y_pred)+lambda2*idcgan_generator_l2_loss(y_true, y_pred)


def idcgan_refined_loss(d_out,vgg_out):
	def loss(y_true, y_pred):
		return lambda1*idcgan_entropy_loss(d_out)+ idcgan_generator_l2_loss(y_true, y_pred) + lambda2*idcgan_perceptual_loss(vgg_out)
	return loss

# ------------------- ResNet9_3D blocks ------------------- #

from keras.layers import Conv3D, Conv3DTranspose, ZeroPadding3D, Cropping3D


class ReflectionPadding3D(Layer):
    
    def __init__(self, padding=(1, 1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=5)]
        super(ReflectionPadding3D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3] + 2 * self.padding[2], s[4])

    def call(self, x, mask=None):
        w_pad, h_pad, d_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [d_pad, d_pad], [0,0] ], 'REFLECT')

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'padding': self.padding
        })
        return config



def ResNet9_3D(opt, use_dropout=False, padding_type='reflect'):
	
	output_nc = opt.output_shape[-1]
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=opt.input_shape)
	padding = "valid"
    
	# ReflectionPad2d(3)
	p1 = ReflectionPadding3D((3,3,3))(in_image)
	c1 = Conv3D(64, kernel_size = (7,7,7), strides = (1,1,1), padding = padding)(p1)
	b1 = BatchNormalization()(c1, training=True) # Training: normalizes wrt current batch input
	a1 = Activation('relu')(b1)
    
	p2 = ZeroPadding3D(padding=(1,1,1))(a1)
	c2 = Conv3D(128, kernel_size = (3,3,3), strides = (2,2,2), padding = padding)(p2) # padding of 1 in Pytorch ???
	b2 = BatchNormalization()(c2, training=True)
	a2 = Activation('relu')(b2)
    
	p3 = ZeroPadding3D(padding=(1,1,1))(a2)
	c3 = Conv3D(256, kernel_size = (3,3,3), strides = (2,2,2), padding = padding)(p3)
	b3 = BatchNormalization()(c3, training=True)
	a3 = Activation('relu')(b3)
    
	r1 = a3 + resnet_block_3D(a3, 256, batchnorm = True, dropout = False)
	r2 = r1 + resnet_block_3D(r1, 256, batchnorm = True, dropout = False)
	r3 = r2 + resnet_block_3D(r2, 256, batchnorm = True, dropout = False)
	r4 = r3 + resnet_block_3D(r3, 256, batchnorm = True, dropout = False)
	r5 = r4 + resnet_block_3D(r4, 256, batchnorm = True, dropout = False)
	r6 = r5 + resnet_block_3D(r5, 256, batchnorm = True, dropout = False)
	r7 = r6 + resnet_block_3D(r6, 256, batchnorm = True, dropout = False)
	r8 = r7 + resnet_block_3D(r7, 256, batchnorm = True, dropout = False)
	r9 = r8 + resnet_block_3D(r8, 256, batchnorm = True, dropout = False)
    
	#x = ZeroPadding2D(padding=(1, 1))(r9)
	x = Conv3DTranspose(128, kernel_size = (3,3,3), strides = (2,2,2), output_padding = 1)(r9)
	x = BatchNormalization()(x, training=True) # Training: normalizes wrt current batch input
	x = Activation('relu')(x)
	x = Cropping3D((1,1,1))(x)
    
	#x = ZeroPadding2D(padding=(1, 1))(x)
	x = Conv3DTranspose(64, kernel_size = (3,3,3), strides = (2,2,2), output_padding = 1)(x) # padding of 1 in Pytorch ???
	x = BatchNormalization()(x, training=True)
	x = Activation('relu')(x)
	x = Cropping3D((1,1,1))(x)
    
	x = ReflectionPadding3D((3,3,3))(x)
	x = Conv3D(output_nc, kernel_size = (7,7,7), strides = (1,1,1), padding = padding)(x)
	#out_image = Activation('sigmoid')(x) # original ResNet has tanh
	if opt.tanh:
		out_image = Activation('tanh')(x)
	else:
		out_image = Activation('sigmoid')(x)
	# define model
	model = Model(in_image, out_image)
	return model

def resnet_block_3D(layer_in, dim, batchnorm = True, dropout = False):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    padding = "valid"
    
    x = ReflectionPadding3D((1,1,1))(layer_in)
    x = Conv3D(dim, kernel_size = (3,3,3), strides = (1,1,1), padding = padding)(x)
    x = BatchNormalization()(x, training = True)
    x = Activation('relu')(x)
    if dropout:
        x = Dropout(0.5)(x, training=True)
    x = ReflectionPadding3D((1,1,1))(x)
    x = Conv3D(dim, kernel_size = (3,3,3), strides = (1,1,1), padding = padding)(x)
    x = BatchNormalization()(x, training = True)
    return x


def Discriminator_3D(opt):#image_shape, output_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=opt.input_shape)
	# target image input
	in_target_image = Input(shape=opt.output_shape)
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64 - 64x64
	d1 = Conv3D(64, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(merged)
    # 64 filters, kernel_size = (4,4), strides, padding = 'same' means the output size is same as input size
    # means that there is even zero-padding left right top bottom.
	d2 = LeakyReLU(alpha=0.2)(d1)
	# C64 2nd - 32x32
	d3 = Conv3D(64, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(d2)
	d4 = BatchNormalization()(d3)
	d5 = LeakyReLU(alpha=0.2)(d4)
	# C128 - 16x16
	d6 = Conv3D(128, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(d5)
	d7 = BatchNormalization()(d6)
	d8 = LeakyReLU(alpha=0.2)(d7)
	# C256 - 8x8
	d9 = Conv3D(256, (4,4,4), strides=(1,1,1), padding='same', kernel_initializer=init)(d8)
	d10 = BatchNormalization()(d9)
	d11 = LeakyReLU(alpha=0.2)(d10)
	# patch output
	d12 = Conv3D(1, (4,4,4), padding='same', kernel_initializer=init)(d11)
	patch_out = Activation('sigmoid')(d12)
	# define model
	if opt.perceptual_loss:
		model = Model([in_src_image, in_target_image], [patch_out, d2, d5, d8, d11])
	else:
		model = Model([in_src_image, in_target_image], patch_out)
	return model

def Discriminator_3D_Koh(opt):#image_shape, output_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=opt.input_shape)
	# C64 - 64x64
	d1 = Conv3D(64, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(in_src_image)
    # 64 filters, kernel_size = (4,4), strides, padding = 'same' means the output size is same as input size
    # means that there is even zero-padding left right top bottom.
	d2 = LeakyReLU(alpha=0.2)(d1)
	# C64 2nd - 32x32
	d3 = Conv3D(64, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(d2)
	#d4 = BatchNormalization()(d3)
	d4 = InstanceNormalization()(d3)
	d5 = LeakyReLU(alpha=0.2)(d4)
	# C128 - 16x16
	d6 = Conv3D(128, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(d5)
	#d7 = BatchNormalization()(d6)
	d7 = InstanceNormalization()(d6)
	d8 = LeakyReLU(alpha=0.2)(d7)
	# C256 - 8x8
	d9 = Conv3D(256, (4,4,4), strides=(1,1,1), padding='same', kernel_initializer=init)(d8)
	#d10 = BatchNormalization()(d9)
	d10 = InstanceNormalization()(d9)
	d11 = LeakyReLU(alpha=0.2)(d10)
	# patch output
	d12 = Conv3D(1, (4,4,4), padding='same', kernel_initializer=init)(d11)
	patch_out = Activation('sigmoid')(d12)
	# define model
	if opt.perceptual_loss:
		model = Model(in_src_image, [patch_out, d2, d5, d8, d11])
	else:
		model = Model(in_src_image, patch_out)
	return model



def mpd_loss(y_true, y_pred):
	#print(y_true.shape)
	#return tf.math.reduce_mean( tf.math.pow( y_true - y_pred, 1.5 ), axis = range(1, len(y_true.shape)) )
	return K.mean( K.pow(y_true-y_pred, 1.5), axis = range(1, len(y_true.shape)) )
	#return tf.math.pow( tf.norm( y_true - y_pred, ord = 1.5, axis = range(1, len(y_true.shape)) ), 1.5 )

"""
from keras.losses import Loss
class mpdLoss(Loss):
  def call(self, y_true, y_pred):
    #loss = tf.math.reduce_mean( tf.math.pow( y_true - y_pred, 1.5 ), axis = range(1, len(y_true.shape)) )
    loss = K.mean( K.pow(y_true-y_pred, 1.5), axis = range(1, len(y_true.shape)) )
	#print(loss.shape)
    return loss
"""




# ALTERNATIVE FROM STACKOVERFLOW

class PackedTensor(tf.experimental.BatchableExtensionType):
    __name__ = 'extension_type_colab.PackedTensor'

    output_0: tf.Tensor
    output_1: tf.Tensor

    # shape and dtype hold no meaning in this context, so we use a dummy
    # to stop Keras from complaining

    shape = property(lambda self: self.output_0.shape)
    dtype = property(lambda self: self.output_0.dtype)

    class Spec:

        def __init__(self, shape, dtype=tf.float32):
            self.output_0 = tf.TensorSpec(shape, dtype)
            self.output_1 = tf.TensorSpec(shape, dtype)

        # shape and dtype hold no meaning in this context, so we use a dummy
        # to stop Keras from complaining
        shape: tf.TensorShape = tf.constant(1.).shape 
        dtype: tf.DType = tf.constant(1.).dtype

@tf.experimental.dispatch_for_api(tf.shape)
def packed_shape(input: PackedTensor, out_type=tf.int32, name=None):
    return tf.shape(input.col_ids)

@tf.experimental.dispatch_for_api(tf.cast)
def packed_cast(x: PackedTensor, dtype: str, name=None):
    return x


class mpdLoss(tf.keras.losses.Loss):
    """ This custom loss function is designed for models with an PackedTensor as
    a single output, so with attributes outputs_0 and outputs_1. This loss will 
    train a model so that outputs_0 represent the predicted class of the input
    image, and outputs_1 will be trained to always be zero (as a dummy). 
    """
    def __init__(self, *args, **kwargs):
        super(mpdLoss, self).__init__(*args, **kwargs)

    def call(self, y_true, y_pred):
        mpd = y_pred - y_true
        mpd = tf.reshape(mpd, (y_pred.shape[0], -1))
        mpd = K.abs(mpd)
        mpd = mpd ** 1.5
        mpd = K.mean(mpd, axis = 1 )
        print(mpd.shape)
        return mpd

# create a layer to combine to pack the outputs in a PackedTensor
class PackingLayer(tf.keras.layers.Layer):
  def call(self, inputs, training=None):
    first_output, second_output = inputs
    packed_output = PackedTensor(first_output, second_output)
    return packed_output
