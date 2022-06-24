import numpy as np


def reduced_centered_range(img, intensity_bg, intensity_root) :
    #Process the img to adjust the value between the 2 intensities, and reverse the black and white
    #Prendre les points de racines -> Fait une mediane ->pareil pour bgd
    maxx = np.amax(img)
    minn = np.amin(img)
    a = (intensity_bg-intensity_root)/(maxx-minn)
    b = intensity_bg-a*maxx
    imgCR = a*img+b
    return np.array(imgCR), a, b



def img_split_no_borders(img : np.array, final_size : tuple, stride) :
    #DEPRECATED
    #Slice img to tiles in final_size shape, and cut the bits that do not fit
    img_w, img_h = img.shape
    tile_w, tile_h = final_size
    tiles=[]
    for i in np.arange(img_w, step=stride):
            for j in np.arange(img_h, step=stride):
                bloc = img[i:i+tile_w, j:j+tile_h]
                if bloc.shape == (final_size[0], final_size[1]):
                    tiles.append(bloc)      
    return tiles


def img_split_remnants(img, tile_size : tuple, stride) :
    #Try to slice img to tiles in final_size shape, and slide the rest to be of the same size
    img = np.array(img)
    img_w, img_h = img.shape
    tile_w, tile_h = tile_size
    tiles=[]
    for i in np.arange(img_w, step=stride):
        for j in np.arange(img_h, step=stride):
            bloc = img[i:i+tile_w, j:j+tile_h]
            if bloc.shape == (tile_w, tile_h):
                tiles.append(bloc)
            else :
                bloc_w, bloc_h = bloc.shape;
                bloc = img[(i-(tile_w-bloc_w)):(i+tile_w), (j-(tile_h-bloc_h)):(j+tile_h)]
                tiles.append(bloc)
    return tiles

#Faire fonction qui prends pas le blocs sans racines ou qui pondere le %tage de bloc sans racines

def img_processing(img : np.array, intensity_bg, intensity_root, tile_size : tuple, stride : int) : 
        img = reduced_centered_range(img, intensity_bg, intensity_root)
        tiles = img_split_remnants(img, tile_size, stride)
        return tiles