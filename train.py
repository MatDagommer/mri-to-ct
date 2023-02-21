# Matthieu Dagommer
# 27/06/2022

# TRAINING A NEW MODEL

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import time
import pickle
from utils import utils
from options.train_options import TrainOptions
from utils.train_function import train_function
from models.networks import *
from models import create_model
from models.custom_model import Custom
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # Removes warning that occur when saving models
from tensorflow import keras
from keras.models import load_model
from keras import backend as K
import pickle
import random as rn


# -------------------- Retrieve Parameters and Data -------------------- #

print("Retrieving Training Parameters and Data...")
# Retrieve training options
opt = TrainOptions().gather_options()

# -------------------- Ensure Reproducibility of the results ------------------- #

#os.environ["PYTHONHASHSEED"] = "0"
keras.utils.set_random_seed(opt.seed)
np.random.seed(opt.seed)

# Force tensorflow to use a single thread
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)

#tf.config.experimental.enable_op_determinism()


# If it doesn't exist yet, create folder to save model
if not os.path.exists(opt.model_path):
    os.mkdir(opt.model_path)


# Retrieve Dataset
dataset_info = pickle.load(open(opt.data_path + opt.dataset + "/dataset_info", "rb"))
train_set = np.load(opt.data_path + opt.dataset + "/train.npz")
valid_set = np.load(opt.data_path + opt.dataset + "/valid_eval.npz")
mm_ct = dataset_info["mm_ct"]


# -------------------- Initialize Model -------------------- #

model = create_model(opt)
model.get_summary(opt)
#exit()

# -------------------- Training -------------------- #

print("Preparing Training...")
start = time.perf_counter()
# TRAINING => Replace with BaseModel.train()
output = train_function(model, train_set, valid_set, opt)
#output = train_function(model, opt)
end = time.perf_counter()
elapsed_time = end - start
print("Total time elapsed: {:.3f} s".format(elapsed_time))


# -------------------- Saving Results -------------------- #

print("Training: Done.\nSaving Training Results...")
pickle.dump(output, open(opt.model_path + 'output_' + opt.name, 'wb'))

# Loading temporary output, in case training was stopped early
temp_output = pickle.load(open(opt.model_path + "temp_output", "rb"))

# Retrieve Mean Absolute Error (MAE) at best epoch
bevel = temp_output[2]
N_valid = len(bevel)
mae = np.absolute(bevel).sum() / N_valid

# Save Model Info (.txt file + dictionary in binary file)
utils.save_info(opt, elapsed_time, mae, mm_ct)
# Learning Curve
utils.learning_curve(opt, temp_output)
# Load Best Model
#model = load_model(opt.model_path + opt.name + '_g.h5')
# Plot Test Samples
#test_set = dataset["test_set"]
#utils.plot_images(model, opt.model_path, test_set, mm_ct)
# Plot histogram
utils.distrib(opt, temp_output)

print("Saving Results: Done")