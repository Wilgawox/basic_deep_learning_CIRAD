from glob import glob

import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime, os
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from IPython.display import clear_output
import tensorflow_addons as tfa
from PIL import Image, ImageSequence

import modif_image
import data_prep_3D

image_path = "Data_Thibault/"
dataset_path = "Data_Thibault/data"
training_data_path = "data_training"
training_data = "training/input/"
val_data_path = "data_validation"
val_data = "training/results/"




#IMG_SIZE = Not for now
N_CHANNELS = 1
N_CLASSES = 2 #softmax
#BATCH_SIZE = ???
#BUFFER_SIZE = ???

#train_set = tf.data.Dataset.list_files(dataset_path + training_data + "*.tif")
#TRAINSET_SIZE = len(glob(dataset_path + val_data + "*.tif"))
#print(f"The Training Dataset contains {TRAINSET_SIZE} images.")

#valid_set = tf.data.Dataset.list_files(dataset_path + training_data + "*.tif")
#VALSET_SIZE = len(glob(dataset_path + val_data + "*.tif"))
#print(f"The Validation Dataset contains {VALSET_SIZE} images.")

#AUTOTUNE = tf.data.experimental.AUTOTUNE

#'''
#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#    try:
#        for gpu in gpus:
#            tf.config.experimental.set_memory_growth(gpu, True)
#        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#    except RuntimeError as e:
#        print(e)'''


#'''----------------------------------------------------------------
#SETUP OF DATASETS
#-----------------------------------------------------------------'''

#dataset = {"train": train_set, "val": valid_set}

#dataset['train'] = dataset['train'].map(data_prep_3D.create_Xarray, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE)
#dataset['train'] = dataset['train'].repeat()
#dataset['train'] = dataset['train'].batch(BATCH_SIZE)
#dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

#dataset['val'] = dataset['val'].map(data_prep_3D.create_Yarray)
#dataset['val'] = dataset['val'].repeat()
#dataset['val'] = dataset['val'].batch(BATCH_SIZE)
#dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

#print(dataset['train'])
#print(dataset['val'])