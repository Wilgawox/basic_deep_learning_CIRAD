#Imports
from audioop import error
from glob import glob
import pandas as pd
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
from skimage.io import imread, imshow
from skimage.transform import resize

# Data paths
training_data = "Data_Thibault/input/"
res_data = "Data_Thibault/results/"
dataset_path = "Data_Thibault/data/"
training_data_path = dataset_path+"data_training/"
val_data_path = dataset_path+"data_validation/"
test_data_path = dataset_path+"data_test"

# Parameters for the tiles :
SIZE = [512,512] #[x,y] dimensions of a tile
IMG_H = SIZE[0]
IMG_W = SIZE[1]
STRIDE =  450 #Stride between two tiles in a image
INT_ROOT = -1 #Value of a root pixel
INT_BG = 1 #Value of a backgroung pixel

#Table de permutation -> Classification
#18/9/9

list_X = []
for filename in glob(training_data+'*.tif'):
    im=Image.open(filename)
    list_X.append(data_prep_3D.create_Xarray(im))

list_Y = []
for filename in glob(res_data+'*.tif'):
    im=Image.open(filename)
    list_Y.append(data_prep_3D.create_Yarray(im))


def image_writing(list_X, list_Y) : 
   for i in range(1,2):#in range(list(list_X)) :
        for j in range(len(list_X[i])):
            tilesX = modif_image.img_processing(list_X[i][j], -1, 1, [512,512], 450)
            tilesY = modif_image.img_processing(list_Y[i][j], -1, 1, [512,512], 450)
            if len(tilesX)!=len(tilesY) :
                raise Exception('Problem while cutting tiles', 'Different number of tiles')
            if i<=len(list_X)//2 :
                #50% of images in training
                np.save((training_data_path+'training/ML1_input_'+str(i)+'.'+str(j)), tilesX)
                np.save((training_data_path+'training/ML1_result_'+str(i)+'.'+str(j)), tilesY)
            elif i>=3*len(list_X)//4 :
                #25% of images in test
                np.save((test_data_path+'test/ML1_input_'+str(i)+'.'+str(j)), tilesX)
                np.save((test_data_path+'test/ML1_result_'+str(i)+'.'+str(j)), tilesY)
            else :
            #25% of images in validation
                np.save((val_data_path+'val/ML1_input_'+str(i)+'.'+str(j)), tilesX)
                np.save((val_data_path+'val/ML1_result_'+str(i)+'.'+str(j)), tilesY)

image_writing(list_X, list_Y);

N_CHANNELS = 1 
N_CLASSES = 2 #softmax #Root or background
NUM_TEST_IMAGES = 0 #Calcul a faire avant execution

#img_list = os.listdir('../Data_Thibault/data/data_test')
#mask_list = os.listdir('../input/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth')

#Creation of a dataset
df_images = pd.DataFrame(img_list, columns=['image_id'])

def get_num_cells(x):
    # split on the _
    a = x.split('_')
    # choose the third item
    b = a[2] # e.g. C53
    # choose second item onwards and convert to int
    num_cells = int(b[1:])
    
    return num_cells

# create a new column called 'num_cells'
#df_images['num_cells'] = df_images['image_id'].apply(get_num_cells)
