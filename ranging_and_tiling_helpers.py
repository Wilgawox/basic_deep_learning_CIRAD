import numpy as np
import paths
import statistics


def reduced_centered_range(img, intensity_bg, intensity_root) :
    #Process the img to adjust the value between the 2 intensities, and reverse the black and white
    maxx = np.amax(img)
    minn = np.amin(img)
    a = (intensity_bg-intensity_root)/(maxx-minn)
    b = intensity_bg-a*maxx
    imgCR = a*img+b
    return np.array(imgCR).astype(np.dtype('float32'))

def average_range(img, intensity_bg, intensity_root) :
    #Take an image to fing its max/min values, take the adjacent pixels, and then find the average value of a pixel of root/background
    #Then it change the image to have max(img)=average_root_value/min(img)=average_min_value
    # NOT YET TESTED
    maxx = np.amax(img)
    max_loc = np.where(img == maxx)#Location of the max value
    max_loc = list(zip(max_loc[0], max_loc[1]))[0]# Tuple with location of max value
    minn = np.amin(img)
    min_loc = list(zip(min_loc[0], min_loc[1]))[0]# Tuple with location of min value
    maxx = statistics.mean([img[max_loc[0][0]][max_loc[1][0]+1]+img[max_loc[0][0]][max_loc[0][1]-1]+img[max_loc[0][0]+1][max_loc[0][1]]+img[max_loc[0][0]-1][max_loc[0][1]]])
    minn = statistics.mean([img[min_loc[0][0]][min_loc[1][0]+1]+img[min_loc[0][0]][min_loc[0][1]-1]+img[min_loc[0][0]+1][min_loc[0][1]]+img[min_loc[0][0]-1][min_loc[0][1]]])
    a = (intensity_bg-intensity_root)/(maxx-minn)
    b = intensity_bg-a*maxx
    imgCR = a*img+b
    return np.array(imgCR).astype(np.dtype('float32'))



def tiling_only_roots(img : np.array, final_size : tuple, stride) :
    #Slice img to tiles in final_size shape, and keep only the slices that have white pixel in it, not using most of the tiles without root in them
    # NOT YET TESTED
    img_w, img_h = img.shape
    tile_w, tile_h = final_size
    tiles=[]
    for i in np.arange(img_w, step=stride):
        for j in np.arange(img_h, step=stride):
            bloc = img[i:i+tile_w, j:j+tile_h]
            if np.max(bloc)>(paths.INT_ROOT+paths.INT_BG)/2 :
                if bloc.shape == (tile_w, tile_h):
                    tiles.append(bloc)
                else :
                    bloc_w, bloc_h = bloc.shape;
                    bloc = img[(i-(tile_w-bloc_w)):(i+tile_w), (j-(tile_h-bloc_h)):(j+tile_h)]
                    tiles.append(bloc)
    return tiles

def tiling(img, final_size : tuple, stride) :
    #Try to slice img to tiles in final_size shape, and slide the rest to be of the same size
    img_w, img_h = img.shape
    tile_w, tile_h = final_size
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

def img_tile_and_range(img : np.array, intensity_bg, intensity_root, tile_size : tuple, stride : int) : 
        img = reduced_centered_range(img, intensity_bg, intensity_root)
        tiles = tiling(img, tile_size, stride)
        return tiles


def reverse_tiling(img_size, tiles, stride) :    
    # This function take an image and its tile list created with the function tiling()
    # We need to keep the same stride used in tiling()
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