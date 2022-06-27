import numpy as np


def reduced_centered_range(img, intensity_bg, intensity_root) :
    #Process the img to adjust the value between the 2 intensities, and reverse the black and white
    #TODO : Prendre les points de racines -> Fait une mediane ->pareil pour bgd
    maxx = np.amax(img)
    minn = np.amin(img)
    a = (intensity_bg-intensity_root)/(maxx-minn)
    b = intensity_bg-a*maxx
    imgCR = a*img+b
    return np.array(imgCR).astype(np.dtype('float32'))



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


def tiling(img, tile_size : tuple, stride) :
    #Try to slice img to tiles in final_size shape, and slide the rest to be of the same size
    # TODO : #Faire fonction qui prends pas le blocs sans racines ou qui pondere le %tage de bloc sans racines
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

def img_processing(img : np.array, intensity_bg, intensity_root, tile_size : tuple, stride : int) : 
        img = reduced_centered_range(img, intensity_bg, intensity_root)
        tiles = tiling(img, tile_size, stride)
        return tiles


def reverse_tiling(img_size, tiles, stride) :    
    # This function take an image and its tile list created with the function tiling()
    # We need to keep the same stride used intiling()
    img_w, img_h = img_size
    tile_w, tile_h = tiles[0].shape
    
    # Creation of 2 variables containing the x/y position of the tiles in the final image
    coordsX=[np.arange(img_w, step=stride)]
    coordsY=[np.arange(img_h, step=stride)]
    
    ntX = len(coordsX[0])
    ntY = len(coordsY[0])
    
    # Adjusting the last coordinate of coordX/coordY
    if(coordsX[0][ntX-1]+tile_w>img_w):
        coordsX[0][ntX-1]=img_w-tile_w
    if(coordsY[0][ntY-1]+tile_h>img_h):
        coordsY[0][ntY-1]=img_h-tile_h
    
    final_img = np.zeros([img_w, img_h])
    
    # Creating a variable to limitate side effects
    semi_overlap_x=(tile_w-stride)//2
    semi_overlap_y=(tile_h-stride)//2

    for i in range(ntX):
        for j in range(ntY):
            cX=coordsX[0][i]
            cY=coordsY[0][j]

            #Identify coordinates into target image
            x0=cX+int(i>0)*semi_overlap_x
            y0=cY+int(j>0)*semi_overlap_y
            xf=cX+tile_w-int(i<(ntX-1))*semi_overlap_x
            yf=cY+tile_h-int(j<(ntY-1))*semi_overlap_y

            #Identify coordinates into source tile
            a0=0+int(i>0)*semi_overlap_x
            b0=0+int(j>0)*semi_overlap_y
            af=tile_w-int(i<(ntX-1))*semi_overlap_x
            bf=tile_h-int(j<(ntY-1))*semi_overlap_y

            tile=tiles[i*ntY+j]
            final_img[x0:xf,y0:yf]=tile[a0:af,b0:bf]
    return final_img