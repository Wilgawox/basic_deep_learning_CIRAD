# Data paths
training_data = "Data_Thibault/input/"
res_data = "Data_Thibault/results/"
dataset_path = "Data_Thibault/data/"
training_data_path = dataset_path+"data_training/"
val_data_path = dataset_path+"data_validation/"
test_data_path = dataset_path+"data_test/"
MODEL_FILEPATH = "Data_Thibault/model.h5"

# Parameters for the tiles
TILE_SIZE = [512,512] #[x,y] dimensions of a tile
IMG_H = TILE_SIZE[0]
IMG_W = TILE_SIZE[1]
STRIDE =  450 #Stride between two tiles in a image
INT_ROOT = -1 #Target value of a root pixel
INT_BG = 1 #Target value of a backgroung pixel

# Parameters for the CNN :
PERCENT_TRAIN_IMAGES = 50 #Calcul a faire avant execution
PERCENT_VALID_IMAGES = 25
PERCENT_TEST_IMAGES = 25
SAMPLE_WEIGHT = 60
PATIENCE = 5
nb_epochs = 200
batch_size = 16
validation_split = 0.1
layers = 3 #Number of layers for the CNN
N_CHANNELS = 1 
N_CLASSES = 2 #softmax #Root or background
NUM_TEST_IMAGES = 0 #Calcul a faire avant execution
#18/9/9