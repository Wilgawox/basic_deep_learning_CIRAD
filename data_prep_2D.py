
import numpy as np
import modif_image
import paths

# Parameters for the tiles :
SIZE = [512,512] #[x,y] dimensions of a tile
IMG_H = SIZE[0] #Height of a tile
IMG_W = SIZE[1] # Weight of a tile
STRIDE =  450 #Stride between two tiles in a image
INT_ROOT = -1 #Value of a root pixel
INT_BG = 1 #Value of a backgroung pixel
CHANNEL = 1 #Images are B&W so 1 channel, would be 3 for RGB
N_CLASSES = 2 #Pixels either belong to the background or to a root

def data_prep_2D_classified(list_X, list_Y) :
    #Function not working anymore, keeping it here for now if I need it later on
    # Process and save the images, putting them in different folders to serve differents puposes
   for i in range(0, len(list_X)) :
        for j in range(len(list_X[i])):
            tilesX = modif_image.data_range_and_tile(list_X[i][j], INT_ROOT, INT_BG, SIZE, STRIDE)
            tilesY = modif_image.tiling(list_Y[i][j], SIZE, STRIDE)
            if len(tilesX)!=len(tilesY) :
                # Check for different size tiles, which would be a big problem later on
                raise Exception('Problem while cutting tiles', 'Different number of tiles')
            for k in range(len(tilesX)):
                if i<=len(list_X)//2 :
                    #We put 50% of images in training ...
                    np.save((paths.training_data_path+'input/ML1_input_img'+str(i)+'.time'+str(j)+'.number'+str(k)), tilesX[k])
                    np.save((paths.training_data_path+'results/ML1_result_img'+str(i)+'.time'+str(j)+'.number'+str(k)), tilesY[k])
                elif i>3*len(list_X)//4 :
                    #... then 25% of images in test ...
                    np.save((paths.test_data_path+'input/ML1_input_img'+str(i)+'.time'+str(j)+'.number'+str(k)), tilesX[k])
                    np.save((paths.test_data_path+'results/ML1_result_img'+str(i)+'.time'+str(j)+'.number'+str(k)), tilesY[k])
                else :
                    #... and the last 25% in validation
                    np.save((paths.val_data_path+'input/ML1_input_img'+str(i)+'.time'+str(j)+'.number'+str(k)), tilesX[k])
                    np.save((paths.val_data_path+'results/ML1_result_img'+str(i)+'.time'+str(j)+'.number'+str(k)), tilesY[k])



def data_prep_2D(list_X, list_Y) : 
    # Save X and Y as .npy in the dataset path, tiled and ranged
    for i in range(0, len(list_X)) :
        for j in range(len(list_X[i])):
            # Ranging and tiling the images
            tilesX = modif_image.data_range_and_tile(list_X[i][j], INT_ROOT, INT_BG, SIZE, STRIDE)
            tilesY = modif_image.tiling(list_Y[i][j], SIZE, STRIDE)
            if len(tilesX)!=len(tilesY) :
                # Check for different size tiles, which would be a big problem later on
                raise Exception('Problem while cutting tiles', 'Different number of tiles')
            for k in range(len(tilesX)):
                np.save((paths.dataset_path+'ML1_input_img'+str(i)+'.time'+str(j)+'.number'+str(k)), tilesX[k])
                np.save((paths.dataset_path+'ML1_result_img'+str(i)+'.time'+str(j)+'.number'+str(k)), tilesY[k])