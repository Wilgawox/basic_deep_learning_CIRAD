#import imageio as iio
#import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence

def create_Xarray(img_2Dt) : 
    X=[]
    for y in ImageSequence.Iterator(img_2Dt):
        X.append(np.array(y))
    return X


def create_Yarray(img_2D) : 
    img_2D=np.array(img_2D)
    Y=[]
    for t in range(1,int(np.max(img_2D)+1)) :
        temp=np.zeros((img_2D.shape[0], img_2D.shape[1]))
        for i in range(len(img_2D)) :
            for j in range(len(img_2D[:,])) :
                p=img_2D[i,j]
                if p!=0 and p<=t : 
                    temp[i,j]=p
        Y.append(temp)
    return Y

