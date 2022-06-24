import paths
import glob
import data_prep_3D
import modif_image
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

    data_prep_2D.data_prep_2D(list_X, list_Y)


dataset_config.CNN(11, 23, 10 )


def catf2D(time_sequence):
    # Input  : a np.array( dim_X, dim_Y, dim_T) with float values indicating probability of background (values near -1) or root (values near 1)
    # Output : a np.array( dim_X, dim_Y ) with integer values indicating for each pixel (x,y) the root apparition time from 1 to max_time, or zero if no_root
    
    N_times=np.shape(time_sequence)[0]

    # The filter_bank is a list of signal models corresponding to apparition of a root, computed for each target time
    filter_bank=np.array([[[[ (j*2-1) if(j<2) else (-1+2*int( i>(j-2)))  for j in range(N_times+1)] ] ] for i in range(N_times)] )  # F Y X T

    # The broadcasted element-wise dotproduct sum(data-mult-bank filter) try all the filters of the bank for each pixel to estimate the likelihood 
    # of a root apparition at each target time. Then we use argmax function to select the index of the filter which gave the highest response
    return np.argmax(np.sum( np.multiply(time_sequence,filter_bank),axis=0),axis=2)


def test_filter_bank():
    # Test of the function on a simple root growing downwards on the third column of the hypermatrix
    N_times=5
    data=np.array([[[[ 1 if (col==2 and lig<=tim) else -1 ] for col in range(N_times+2)] for lig in range(N_times+3)] for tim in range(N_times)] )
    print("Data provided : "+str(np.shape(data)))
    data2d=catf2D_2(data)
    print("Result should show a root growing to the south")
    print(data2d)


