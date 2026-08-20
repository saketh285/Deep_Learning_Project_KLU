import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from pasta.augment import inline
#%matplotlib inline
from keras.preprocessing.image import load_img,img_to_array

pic_size = 48
b_path = "images"
plt.figure(0,figsize=(20,20))
cpt=0
for expression in os.listdir(b_path + "train"):
    for i in range(1,8):
        cpt += 1
        plt.subplot(7,8,cpt)
        img=load_img(b_path+"train"+expression+"/"+os.listdir(b_path+"train"+expression)[i],target_size=(pic_size,pic_size))
        plt.imshow(img,cmap='gray')
        plt.xlabel(os.listdir(b_path+"train"+expression)[i])
plt.tight_layout()
plt.show()