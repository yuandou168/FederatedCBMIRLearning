import tensorflow as tf
import pandas as pd
from my_data_generator import DataGenerator
# from variational_autoencoder import ConvVarAutoencoder
from variational_autoencoder_YD import ConvVarAutoencoder
# from utils_image_retrieval import save_reconstructed_images, create_environment, create_json
from utils_image_retrieval_YD import save_reconstructed_images, create_environment, create_json
#from tensorflow.keras.models import load_model
import os, time, csv
import numpy as np
import cv2
import matplotlib.pyplot as plt 
from PIL import Image,ImageOps,ImageChops
import pickle
from skimage.transform import resize

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session=tf.compat.v1.Session(config=config)

# Hyper-parametrs
input_dim = (224, 224, 3)
encoder_conv_filters = [32, 64, 128, 256]
encoder_conv_kernel_size = [3, 3, 3, 3]
encoder_conv_strides = [1, 2, 2, 2]
bottle_conv_filters = [64, 32, 1, 256]
bottle_conv_kernel_size = [3, 3, 3, 3]
bottle_conv_strides = [1, 1, 1, 1]
decoder_conv_t_filters = [128, 64, 32, 3]
decoder_conv_t_kernel_size = [3, 3, 3, 3]
decoder_conv_t_strides = [2, 2, 2, 1]
# sess = tf.function()
bottle_dim = (32, 32, 256)
z_dim = 200
r_loss_factor = 10000
lr = 0.000001
batch_size = 16
epochs = 300
is_training = True
conv_layeri=[]
conv_t_layeri=[]


round_idx = 1
# I# I/O paths
run_folders = {
   
     "model_path": '/home/yuandou/test/breakhis100x_FLC4/CAE_100X/check/CL.CAE/'
    , "results_path": '/home/yuandou/test/breakhis100x_FLC4/CAE_100X/check/CL.CAE/'
    , "log_filename": '/home/yuandou/test/breakhis100x_FLC4/CAE_100X/check/log_breast.csv'
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
df_pneumo_2d = pd.read_csv(r"/home/yuandou/test/archive/train/x_train_breast100x.csv")
df_pneumo_2d.columns = ['index', 'image_name']


df_pneumo_2d=(df_pneumo_2d.iloc[:,:])

label2 = range(0,len(image_names))
# image_directory="E:/Codes_prostatecancer/Prostate_cancer/SICAPv2/SICAPv2/images3/"
# parasitized_images = os.listdir(image_directory )


#%%################
from sklearn.model_selection import train_test_split   
x_train, x_val, ytrain,ytest=train_test_split(df_pneumo_2d,label2, test_size = 0.2, random_state = 0)
# Saving Training set
# x_train.to_csv (r'C:/Users/Zahra/Downloads/archive/BreaKHis_400X/x_train.csv', index = True, header=True)
# x_val.to_csv (r'C:/Users/Zahra/Downloads/archive/BreaKHis_400X/x_val.csv', index = True, header=True)

#%%
ytrain = x_train.iloc[:,:]
ytrain = ytrain.drop("image_name", 1)  
# ytrain = ytrain.drop("Index", 1)
yval = x_val.iloc[:,:]
yval = yval.drop("image_name", 1)  
# yval = yval.drop("indexs", 1)
if is_training:
    etime2trainmodel = time.time()
    # import pdb; pdb.set_trace()
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
    etime2trainmodel = time.time()
     
#%% Test data generator

df_pneumo_2d_test = pd.read_csv(run_folders["tsv_path_test"])
df_pneumo_2d_test.columns = ['image_name', 'label']

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

duration = etime2trainmodel - stime2trainmodel

header = ["stime2trainmodel", "etime2trainmodel", "duration"]
data = [stime2trainmodel, etime2trainmodel, duration]

dirctory = run_folders["model_path"] + run_folders["exp_name"] 
with open(dirctory+'/res_get_timecost_client4_breaKHis_100x.csv', 'a', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write the data
    writer.writerow(data)
