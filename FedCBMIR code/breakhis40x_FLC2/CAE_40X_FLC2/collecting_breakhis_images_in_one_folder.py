import tensorflow as tf
import pandas as pd
from my_data_generator import DataGenerator
from variational_autoencoder import ConvVarAutoencoder
from utils_image_retrieval import save_reconstructed_images, create_environment, create_json
#from tensorflow.keras.models import load_model
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt 
from PIL import Image,ImageOps,ImageChops
import pickle
from skimage.transform import resize
from PIL import Image,ImageOps
# df_pneumo_2d = pd.read_csv(r"C:\Users\Zahra\Downloads\archive\Folds.csv")
df_pneumo_2d = pd.read_csv(r"/home/yuandou/test/archive/Folds.csv")
df_pneumo_2d.columns = ['fold','mag','grp', 'image_name']
df_pneumo_2d=(df_pneumo_2d.iloc[:,:])
# image = Image.open('C:/Users/Zahra/Downloads/archive/'+ os.path.join(df_pneumo_2d["filename"][i]))
        
# i=0
# for i in range(len(df_pneumo_2d)):

#     image = Image.open('C:/Users/Zahra/Downloads/archive/'+ os.path.join(df_pneumo_2d["image_name"][i]))
#     img = cv2.imread('C:/Users/Zahra/Downloads/archive/'+ os.path.join(df_pneumo_2d["image_name"][i]), 1)
#     cv2.imwrite(r'C:\Users\Zahra\Downloads\archive\40X_images/'+ str((os.path.split(df_pneumo_2d["image_name"][i]))[1]),img)
  
   #%% 
# A = (os.path.split(df_pneumo_2d["image_name"][1]))[1]
#%%
import os
import shutil

# fetch all files
# destination = r'C:\Users\Zahra\Downloads\archive\beaKHis_images/' 
destination = r'/home/yuandou/test/archive/breaKHis_images/'     
for i in range(len(df_pneumo_2d)):
    if df_pneumo_2d["fold"][i] == 1:
        image = '/home/yuandou/test/archive/'+ os.path.join(df_pneumo_2d["image_name"][i])
        shutil.copy(image, destination)


        