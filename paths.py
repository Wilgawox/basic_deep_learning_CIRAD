# Data paths
training_data = "Data_Thibault/input/"
res_data = "Data_Thibault/results/"
dataset_path = "Data_Thibault/data/"
training_data_path = dataset_path+"data_training/"
val_data_path = dataset_path+"data_validation/"
test_data_path = dataset_path+"data_test/"
MODEL_FILEPATH = "Data_Thibault/model.h5"

TILE_SIZE = [512,512] #[x,y] dimensions of a tile
IMG_H = SIZE[0]
IMG_W = SIZE[1]
STRIDE =  450 #Stride between two tiles in a image
INT_ROOT = -1 #Target value of a root pixel
INT_BG = 1 #Target value of a backgroung pixel
N_CHANNELS = 1 
N_CLASSES = 2 #softmax #Root or background