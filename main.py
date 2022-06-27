import paths
import glob
import data_prep_3D
import ranging_and_tiling_helpers
import dataset_config
import data_prep_2D

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


#I1 = Image.open("Data_Thibault\Input\ML1_Boite_00009.tif")
#I2 = Image.open("Data_Thibault\Results\ML1_Boite_00009.tif")

#X = data_prep_3D.create_Xarray(I1)
#Y = data_prep_3D.create_Yarray(I2)
##plt.imshow(X[20])
##plt.show
##plt.imshow(Y[20])
#X = modif_image.raange(X[3],1,-1)


Inp = input("Do you want to create .npy files ? (Y/N)")

if Inp == 'Y' or Inp=='y' : 
    list_X = []
    for filename in glob(paths.training_data+'*.tif'):
        im=Image.open(filename)
        list_X.append(data_prep_3D.create_Xarray(im))

    list_Y = []
    for filename in glob(paths.res_data+'*.tif'):
        im=Image.open(filename)
        list_Y.append(data_prep_3D.create_Yarray(im))

data_prep_2D.data_arborescence_setup(list_Y)


dataset_config.CNN(11, 23, 10 )
# TODO : Un modele - 1 imge : visualiser la sortie 