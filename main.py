import data_prep_3D
import modif_image
import dataset_config
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


I1 = Image.open("Data_Thibault\Input\ML1_Boite_00009.tif")
I2 = Image.open("Data_Thibault\Results\ML1_Boite_00009.tif")

X = data_prep_3D.create_Xarray(I1)
Y = data_prep_3D.create_Yarray(I2)
#plt.imshow(X[20])
#plt.show
#plt.imshow(Y[20])
X = modif_image.raange(X[3],1,-1)
