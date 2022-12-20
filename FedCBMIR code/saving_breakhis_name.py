# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:47:23 2022

@author: Zahra
"""

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import pickle
import pandas as pd
from tensorflow import keras
import tensorflow
import matplotlib.pyplot as plt 
import cv2
from PIL import Image,ImageOps
import os



run_folders1 = {
    "tsv_path_test": r"C:\Users\Zahra\Downloads\archive\BreaKHis_100X_1fold/x_train_val_100X_merge.csv",

}
testX=pd.read_csv(run_folders1["tsv_path_test"])
testX = np.array(testX)
# testX=testX.iloc[:,:]
name = []
img = []
for i in range(0,len(testX)):
    name.append( testX[i][1])
    img.append((os.path.split(name[i]))[-1])
    
img = pd.DataFrame(img)   
img.to_csv (r"C:\Users\Zahra\Downloads\archive\BreaKHis_100X_1fold/x_train_val_100X_name.csv", index = True, header=True)
