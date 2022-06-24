#Imports
from glob import glob
import numpy as np
import tensorflow as tf
import datetime
import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers as tfk 
from keras.callbacks import *
from PIL import Image
import data_prep_3D
import paths

# Data paths
training_data = "Data_Thibault/input/"
res_data = "Data_Thibault/results/"
dataset_path = "Data_Thibault/data/"
training_data_path = dataset_path+"data_training/"
val_data_path = dataset_path+"data_validation/"
test_data_path = dataset_path+"data_test"

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
#18/9/9

N_CHANNELS = 1 
N_CLASSES = 2 #softmax #Root or background
NUM_TEST_IMAGES = 0 #Calcul a faire avant execution

def create_XY(n_img, time, tile_number) :
    X = []
    Y = []
    for i in range(n_img+1) : 
        for t in range(time+1) :
            for n in range(tile_number+1) :
                #if i, t et n < parametres max pour les images
                a = np.load(paths.dataset_path+'ML1_input_img'+str(i)+'.time'+str(t)+'.number'+str(n)+'.npy')
                b = np.load(paths.dataset_path+'ML1_result_img'+str(i)+'.time'+str(t)+'.number'+str(n)+'.npy')
                X.append(a)
                Y.append(b)
    return X, Y



def shuffle_XY(liste_X, liste_Y, percent_train, percent_valid, percent_test) :
    N = len(liste_X)
    random_list = np.arange(N)
    np.random.shuffle(random_list)
    
    X_temp = np.array(liste_X)[random_list]
    Y_temp = np.array(liste_Y)[random_list]
    
    X_train = X_temp[0:(percent_train*N)//100]
    X_valid = X_temp[0:(percent_valid*N)//100]
    X_test = X_temp[0:(percent_test*N)//100]
    
    Y_train = Y_temp[0:(percent_train*N)//100]
    Y_valid = Y_temp[0:(percent_valid*N)//100]
    Y_test = Y_temp[0:(percent_test*N)//100]
    return X_train,Y_train,X_test,Y_test,X_valid,Y_valid



def CNN(n_img, time, tile_number) :
    X,Y = create_XY(n_img, time, tile_number)
    X_train,Y_train,X_test,Y_test,X_valid,Y_valid=shuffle_XY(X, Y, PERCENT_TRAIN_IMAGES, PERCENT_VALID_IMAGES, PERCENT_TEST_IMAGES)
    inputs = tfk.Input(shape=(512, 512, 1))
    convo1 = tfk.Conv2D(batch_size, layers, activation='sigmoid', padding='same', kernel_initializer='he_normal')(inputs)
    #convo1 = tfk.Dropout(0.1)(convo1)
    convo2 = tfk.Conv2D(batch_size, layers, activation='sigmoid', padding='same', kernel_initializer='he_normal')(convo1)

    output = tfk.Conv2D(1, 1, activation = 'sigmoid')(convo2)

    model = tf.keras.Model(inputs = inputs, outputs = output)
    model.compile(optimizer='rmsprop',
        loss=keras.losses.BinaryCrossentropy(),
        metrics=['accuracy'])

    filepath = paths.MODEL_FILEPATH

    # Calculate the weights for each class so that we can balance the data
    #print(np.shape(sample_weight))
    #print(np.shape(X_train), np.shape(Y_train))
    
    # Add the class weights to the training                                         
    sample_weight = np.ones(np.shape(Y_train))
    sample_weight[Y_train == 1] = SAMPLE_WEIGHT

    
    earlystopper = EarlyStopping(patience=PATIENCE, verbose=1)

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                                 save_best_only=True, mode='min')
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks_list = [tensorboard_callback,earlystopper, checkpoint]

    history = model.fit(np.array(X_train), 
                        np.array(Y_train), 
                        validation_split=validation_split, batch_size=batch_size, 
                        epochs=nb_epochs, callbacks=callbacks_list)

    model.load_weights(paths.MODEL_FILEPATH)
    test = model.predict(X_test)
    test_img_pred = test[0, :, :, 0]
    test_image = X_test[0, :, :]
    test_res_img = Y_test[0, :, :]
    plt.figure(figsize=(10,10))

    plt.subplot(1,3,1)
    plt.imshow(test_image, cmap='gray')
    plt.title('Original', fontsize=14)
    plt.subplot(1,3,2)

    plt.imshow(test_img_pred, cmap='gray')
    plt.title('Predicted', fontsize=14)
    
    plt.subplot(1,3,3)
    plt.imshow(test_res_img, cmap='gray')
    plt.title('Result we want', fontsize=14)