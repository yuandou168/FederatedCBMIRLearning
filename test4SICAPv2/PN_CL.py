# from numba import cuda 
# device = cuda.get_current_device()
# device.reset()
#%%
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
# Hyper-parametrs
input_dim = (128, 128, 3)
encoder_conv_filters = [64, 128, 256]
encoder_conv_kernel_size = [3, 3, 3]
encoder_conv_strides = [1, 2, 2]
bottle_conv_filters = [64, 32, 1, 256]
bottle_conv_kernel_size = [3, 3, 3, 3]
bottle_conv_strides = [1, 1, 1, 1]
decoder_conv_t_filters = [128, 64, 3]
decoder_conv_t_kernel_size = [3, 3, 3]
decoder_conv_t_strides = [2, 2, 1]
# sess = tf.function()
bottle_dim = (32, 32, 256)
z_dim = 200
r_loss_factor = 10000
lr = 0.0005
batch_size = 8
epochs = 25
is_training = True
conv_layeri=[]
conv_t_layeri=[]


round_idx = 1

# I# I/O paths
run_folders = {
    # "tsv_path": "/home/yuandou/test/CBIR/SICAPv2/partition/Test/Train1.csv"
    "tsv_path": "/home/yuandou/test/CBIR/SICAPv2/x_train_FDL_1.csv"
    # "tsv_path": "/home/yuandou/test/CBIR/SICAPv2/Train1.csv"
    # ,"tsv_path_val": "E:/Codes_prostatecancer/Prostate cancer/SICAPv2/SICAPv2/partition/Test/Test.csv"
    ,"tsv_path_test":"/home/yuandou/test/CBIR/SICAPv2/partition/Test/Test.csv"
    , "data_path": "/home/yuandou/test/CBIR/SICAPv2/images"
    , "model_path": '/home/yuandou/test/CBIR/check/Centralized.CAE.VAE.beforeTSNE/'
    , "results_path": '/home/yuandou/test/CBIR/check/Centralized.CAE.VAE.beforeTSNE/'
    , "log_filename": '/home/yuandou/test/CBIR/check/Centralized.CAE.VAE.beforeTSNE/prostate_cancer.csv'
}
# Creating the required folders
create_environment(run_folders)


# Building JSON with the model hyperparameters
hyperparameters = {
    "input_dim": input_dim
    , "encoder_conv_filters": encoder_conv_filters
    , "encoder_conv_kernel_size": encoder_conv_kernel_size
    , "encoder_conv_strides": encoder_conv_strides
    , "bottle_conv_filters" : bottle_conv_filters
    , "bottle_conv_kernel_size": bottle_conv_kernel_size
    , "bottle_conv_strides":  bottle_conv_strides
    , "decoder_conv_t_filters": decoder_conv_t_filters
    , "decoder_conv_t_kernel_size": decoder_conv_t_kernel_size
    , "decoder_conv_t_strides": decoder_conv_t_strides
    , "z_dim": z_dim
    , "r_loss_factor": r_loss_factor
    , "learning_rate": lr
    , "batch_size": batch_size
    , "epochs": epochs
    , "opt": "Adam"
    , "loss_function": "mse"
    , "data_path": run_folders["data_path"]
    , "bottle_dim" : bottle_dim 
}
create_json(hyperparameters, run_folders)

label2=[]
df_pneumo_2d = pd.read_csv(run_folders["tsv_path"])
# df_pneumo_2d.columns = ['image_name', 'NC', 'G3', 'G4', 'G5'] # with Train1.csv
df_pneumo_2d.columns = ['index', 'image_name', 'NC', 'G3', 'G4', 'G5'] # with *_FDL.csv

df_pneumo_2d=(df_pneumo_2d.iloc[:,:])
# index=df_pneumo_2d["index"]
image_names=df_pneumo_2d["image_name"]
dataset = []
label2 = range(0,len(image_names))


#%%################
from sklearn.model_selection import train_test_split   
# x_train, x_val, ytrain,ytest=train_test_split(df_pneumo_2d,label2, test_size = 0.2, random_state = 0) #0.2 --> 20% dataset for validation; 80% for training
x_train, x_val, ytrain, ytest=train_test_split(df_pneumo_2d,label2, test_size = 0.1, random_state = 0) #0.1 --> 10% dataset for validation; 90% for training
# Saving Training set
x_train.to_csv (r'/home/yuandou/test/CBIR/SICAPv2/x_train_CL.csv', index = True, header=True) # 90% of x_train_FDL.csv for training
x_val.to_csv (r'/home/yuandou/test/CBIR/SICAPv2/x_val_CL.csv', index = True, header=True) # 10% of x_train_FDL.csv for validation


ytrain = x_train.iloc[:,:]
ytrain = ytrain.drop("image_name", 1)  
# ytrain = ytrain.drop("Index", 1)
yval = x_val.iloc[:,:]
yval = yval.drop("image_name", 1)  
# yval = yval.drop("indexs", 1)


""" Train Model"""
if is_training:

    # import pdb; pdb.set_trace()
    """Load Data""" 
    data_flow_train = DataGenerator(x_train
                                    , input_dim[0]
                                    , input_dim[1]
                                    , input_dim[2]
                                    , indexes_output=[True, True, False, False]
                                    , batch_size=batch_size
                                    , path_to_img=run_folders["data_path"]
                                    , data_augmentation=True
                                    , vae_mode=True
                                    , reconstruction=True
                                    , softmax=False
                                    , hide_and_seek=False
                                    , equalization=False
                                    )

    data_flow_dev = DataGenerator(x_val
                                  , input_dim[0]
                                  , input_dim[1]
                                  , input_dim[2]
                                  , indexes_output=[True, True, False, False]
                                  , batch_size=batch_size
                                  , path_to_img=run_folders["data_path"]
                                  , data_augmentation=True
                                  , vae_mode=True
                                  , reconstruction=True
                                  , softmax=True
                                  , hide_and_seek=False
                                  , equalization=False
                                  )

    # VAE instance
    """ Architecture"""
    my_VAE = ConvVarAutoencoder(input_dim
                 , encoder_conv_filters
                 , encoder_conv_kernel_size
                 , encoder_conv_strides
                 , bottle_dim
                 , bottle_conv_filters
                 , bottle_conv_kernel_size
                 , bottle_conv_strides
                 , decoder_conv_t_filters
                 , decoder_conv_t_kernel_size
                 , decoder_conv_t_strides
                 , z_dim)

    # Buildig VAE
    # import pdb; pdb.set_trace()
    my_VAE.build(use_batch_norm=True, use_dropout=True)
    # print(my_VAE.model.summary())

    # Compiling VAE
    my_VAE.compile(learning_rate=lr, r_loss_factor=r_loss_factor)
    print(my_VAE.model.summary())
    # Training VAE
    steps_per_epoch = len(data_flow_train)
    H = my_VAE.train_with_generator(data_flow_train, epochs, steps_per_epoch, data_flow_dev, run_folders, round_idx)
    # my_VAE.save_model(run_folders, H)
     
#%% Test data generator

df_pneumo_2d_test = pd.read_csv(run_folders["tsv_path_test"])
df_pneumo_2d_test.columns = ['image_name', 'NC', 'G3', 'G4', 'G5']

x_test=df_pneumo_2d_test.iloc[:,:]
data_generator_test = DataGenerator(x_test
                                    , input_dim[0]
                                    , input_dim[1]
                                    , input_dim[2]
                                    , indexes_output=[True, True, False, False]
                                    , batch_size=batch_size
                                    , path_to_img=run_folders["data_path"]
                                    , data_augmentation=True
                                    , vae_mode=True
                                    , reconstruction=True
                                    , softmax=False
                                    , hide_and_seek=False
                                    , equalization=False
                                    )
