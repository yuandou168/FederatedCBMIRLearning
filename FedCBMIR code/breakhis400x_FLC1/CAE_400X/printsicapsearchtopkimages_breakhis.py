##
#GPU

# 
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session=tf.compat.v1.Session(config=config)
##
from sklearn.utils.class_weight import compute_class_weight
import math
from sklearn.utils.multiclass import unique_labels
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
import cv2
from PIL import Image,ImageOps
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import math
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import csv

import argparse
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--id', action='store', type=str, required=True, dest='id')


arg_parser.add_argument('--exp', action='store', type=str, required='True', dest='exp')
arg_parser.add_argument('--round', action='store', type=str, required='True', dest='round')
# arg_parser.add_argument('--index', action='store', type=int, required='True', dest='index')
arg_parser.add_argument('--top_k', action='store', type=int, required='True', dest='top_k')


args = arg_parser.parse_args()

id = args.id

root = r"/home/yuandou/test/breakhis400x_FLC1/CAE_400X/check/FL.CAE/"

exp_path = root + args.exp +"/" 
# exp_path = r"/home/yuandou/test/breakhis40x_FLC2/CAE_40X_FLC2/check/FL.CAE/exp_0001/"
weights_path = exp_path + args.round + "/weights/VAE_weights.h5"
model_path = exp_path + args.round + "/VAE_model.pkl"
index_path = exp_path + "SE/index_FLC2_400x_224.csv"

top_k = args.top_k

trainX_datacsv = '/home/yuandou/test/archive/train/Merge_Train_val_name400X.csv'
testX_datacsv = '/home/yuandou/test/archive/test/x_test400X_name.csv'
image_path = '/home/yuandou/test/archive/breaKHis_images/'

testX_labelscsv = r'/home/yuandou/test/archive/test/x_test_breast400x_label.csv'
trainX_labelscsv = r'/home/yuandou/test/archive/train/Merge_Train_val_name400x_label.csv'

cm_output = exp_path + "SE/ConfusionMatrixDisplay_top"+str(top_k)+".png"
ev_output = exp_path + "SE/ModelMetricDisploy_top"+str(top_k)+".csv"
sm_output = exp_path + "SE/SimilarImageDisploy_top"+str(top_k)+".png"
#Directions
args1 = {
	# "index": r"C:\Users\Zahra\Downloads\archive\BreaKHis_40X_1fold/index128.csv"
    # "index": r"/home/yuandou/test/breakhis40x_FLC2/CAE_40X_FLC2/check/FL.CAE/exp_0002/index224.csv"
    "index": index_path

}
run_folders1 = {
    # "weights_path":r"C:/Users/Zahra/Downloads/archive/BreaKHis_40X_1fold/exp_0023/weights/VAE_weights.h5"
    # "weights_path": r"/home/yuandou/test/breakhis40x_FLC2/CAE_40X_FLC2/check/FL.CAE/exp_0002/round_0005/weights/VAE_weights.h5"
    "weights_path": weights_path
}

#Reading Data
batch_size = 1
backbone = 'fsconv'
classes = ['image_name']
input_shape = (224, 224, 3)
data_generator_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, featurewise_center= False,
                                                                       samplewise_center=False,
                                                                       featurewise_std_normalization=False,
                                                                       samplewise_std_normalization=False,
                                                                       zca_whitening=False,
                                                                       zca_epsilon=1e-06,
                                                                       rotation_range=90,
                                                                       width_shift_range=0.05,
                                                                       height_shift_range=0.05,
                                                                       brightness_range=[0.5, 1.5],
                                                                       shear_range=0.0,
                                                                       zoom_range=0.0,
                                                                       channel_shift_range=0.0,
                                                                       fill_mode='nearest',
                                                                       cval=0.0,
                                                                       horizontal_flip=True,
                                                                       vertical_flip=True,
                                                                       preprocessing_function=None)

# trainX = data_generator_train.flow_from_dataframe(dataframe=pd.read_csv('/home/yuandou/test/archive/train/Merge_Train_val_name40x.csv'),
                                                #  directory='/home/yuandou/test/archive/breaKHis_images/',
trainX = data_generator_train.flow_from_dataframe(dataframe=pd.read_csv(trainX_datacsv),
                                                 directory=image_path,
                                                 x_col='image_name',
                                                 y_col=classes,
                                                 batch_size=batch_size,
                                                 seed=42,
                                                 shuffle=True,
                                                 class_mode='raw',
                                                 target_size=input_shape[:-1])

data_generator_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
# testX = data_generator_test.flow_from_dataframe(dataframe=pd.read_csv('/home/yuandou/test/archive/test/x_test40X_name.csv'),
                                            #    directory='/home/yuandou/test/archive/breaKHis_images/',
data_generator_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
testX = data_generator_test.flow_from_dataframe(dataframe=pd.read_csv(testX_datacsv),
                                               directory=image_path,
                                               x_col='image_name',
                                               y_col=classes,
                                               batch_size=batch_size,
                                               seed=42,
                                               shuffle=False,
                                               class_mode='raw',
                                               target_size=input_shape[:-1])


#%% load our autoencoder from disk
print("[INFO] loading autoencoder model...")
from variational_autoencoder import ConvVarAutoencoder
# import pdb; pdb.set_trace()
def load_model_AD1(self):
        # with open('/home/yuandou/test/breakhis40x_FLC2/CAE_40X_FLC2/check/FL.CAE/exp_0002/round_0005/VAE_model.pkl', 'rb') as f:
        with open(model_path, 'rb') as f:
            params = pickle.load(f)
        my_VAE = ConvVarAutoencoder(*params)
        my_VAE.build(use_batch_norm=True, use_dropout=True)
        my_VAE.model.load_weights(os.path.join(os.path.join(run_folders1["weights_path"])))
        return my_VAE

autoencoder = load_model_AD1("VAE_model.pkl")
print(autoencoder.model.summary())

FE = Model(inputs=autoencoder .model.get_layer("encoder_input").input,
 	outputs=autoencoder.model.get_layer("encoder_output").output)
print(FE.summary())
#%%##################

#Feature Extractor
features = FE.predict(trainX, math.ceil(trainX.n / batch_size))
# image to its corresponding latent-space representation
indexes = list(range(0, len(trainX)))
data = {"indexes": indexes, "features": features}
# write the data dictionary to disk
print("[INFO] saving index...")
f = open(args1["index"], "wb")
f.write(pickle.dumps(data))
f.close()

#%%#####################
#CON2
from scipy.spatial import distance
from scipy.spatial.distance import cityblock, cosine
def euclidean(a, b):
	euclidean = np.linalg.norm(a - b) #cosine_similarity = 1 - cosine(a, b)
    #distance1= distance.hamming(a,b) #euclidean = np.linalg.norm(a - b) #manhattan=distance.cityblock(a,b) # compute and return the euclidean distance between two vectors
	return euclidean

def perform_search(queryFeatures, index, maxResults=5):
	# initialize our list of results
	results = []
	# loop over our index
	for i in range(0, len(index["features"])):
		# compute the euclidean distance between our query features
		# and the features for the current image in our index, then
		# update our results list with a 2-tuple consisting of the
		# computed distance and the index of the image
		d = euclidean(queryFeatures, index["features"][i])
		results.append((d, i))
	# sort the results and grab the top ones
	results = sorted(results)[ : maxResults]
	# return the list of results
	return results; 
# construct the argument parse and parse the arguments
# results = sorted(results)[ : maxResults]

#%%
# tuning the number of results and query images
from sklearn.metrics import top_k_accuracy_score
index = pickle.loads(open(args1["index"], "rb").read())
features_test = FE.predict(testX, math.ceil(testX.n / batch_size))

maxResults = top_k
# maxResults = 3
# size = 25
# loading indexs
print("HERE")
#%%
list_similar = []
testlist = []
query_name = []
images1 = [];
list_query = []

# testX_labels = pd.read_csv(r'C:/Users/Zahra/Downloads/archive/BreaKHis_400X_1fold/x_test_breast400x_label.csv') 
# trainX_labels = pd.read_csv(r'C:/Users/Zahra/Downloads/archive/BreaKHis_400X_1fold/Merge_Train_val_name400x_label.csv') 
testX_labels = pd.read_csv(testX_labelscsv) 
trainX_labels = pd.read_csv(trainX_labelscsv)
testX_labels.columns = ['label']
trainX_labels.columns = ['label']
testX_labels1=(testX_labels.iloc[:,:])
trainX_labels1=(trainX_labels.iloc[:,:])

#Reading Data
args1 = {
	# "index": r"C:\Users\Zahra\Downloads\archive\BreaKHis_400X_1fold/index128.csv"
    "index": index_path
}
run_folders2 = {
    # "weights_path":r"C:\Users\Zahra\Downloads\archive\BreaKHis_400X_1fold\GPU\exp_0003/weights/VAE_weights.h5"
    "weights_path": weights_path
    
}
TestName = []
trainName = []
# trainName = pd.read_csv(r"C:\Users\Zahra\Downloads\archive\BreaKHis_400X_1fold/Merge_Train_val_name400X.csv")
trainName = pd.read_csv(trainX_datacsv)
trainName.columns = ['image_name_train']
trainName = trainName.iloc[:,:]
# TestName = pd.read_csv(r"C:\Users\Zahra\Downloads\archive\BreaKHis_400X_1fold/x_test400X_name.csv")
TestName = pd.read_csv(testX_datacsv)
TestName.columns = ['image_name_test']
# TestName = TestName.iloc[:,:]
TestName = TestName["image_name_test"]
TrainName = trainName["image_name_train"]
list_similar_name = []
list_query_name = []
import random
# resultant random numbers list
queryIdxs=[]
# traversing the loop 15 times
for i in range(15):
   # generating a random number in the range 1 to 100
   r=random.randint(0, len(TestName))
   # checking whether the generated random number is not in the
   # randomList
   if r not in queryIdxs:
      # appending the random number to the resultant list, if the condition is true
      queryIdxs.append(r)
# printing the resultant random numbers list
print("non-repeating random numbers are:")
print(queryIdxs)

# queryIdxs = random.randint(0, len(TestName))
#searching and finding results      
for i in queryIdxs:
    queryFeatures = features_test[i]
    results = perform_search(queryFeatures, index , maxResults)    
    #loop over the Results    
    for (d , j) in results:
        # import pdb; pdb.set_trace()
        list_similar.append(trainX.labels[j])
        list_similar_name.append(TrainName[j])
    list_query.append(testX.labels[i])
    list_query_name.append(TestName[i])
    #similar images
    
######
#%%plotting  
# setting values to rows and column variables
rows = 2
r = 1
d = 0
X = [ (2,1,1), (2,3,4), (2,3,5), (2,3,6) ]
maxResults = top_k
columns = maxResults
# image_directory = r"C:\Users\Zahra\Downloads\archive\beaKHis_images/"
image_directory = image_path
# reading images
b = 400 
cc = maxResults
for i in range(0,  len(queryIdxs)):
    # if i == 16:
        Image_query = plt.imread(image_directory+list_query_name[i])
        # 
        # Adds a subplot at the 1st position
        fig = plt.figure()
        for j in range (b , cc):
            # d = 1
            d += 1
            f += 1
            print(d)
            Image_similar = plt.imread(image_directory+list_similar_name[j])
                    #query 
            ax2 = fig.add_subplot(2, columns, 1)      
            ax2.imshow(Image_query)
            plt.axis('off')
            plt.title("Query")
            #similar
            # import pdb; pdb.set_trace()
            ax1 = fig.add_subplot(2, columns, d)
            ax1.imshow(Image_similar)
            plt.axis('off')
            plt.title("Retrived"+" : " + str(f))
    
            r += 1
        d = 3
        f = 1
        b = cc
        cc = (i+2)*maxResults
        r = 1
        fig.savefig(sm_output)
#evaluate
# # Reset data generator after prediction and set references
# preds = np.argmax(list_similar, axis=1)  # Predictions (Label)
# refs = np.argmax(list_query, axis=1)  # References
# end_list_similar = []
# flage = 0
# b = 0
# i = 0
# f = 0
# cc = maxResults
# nothing = np.array([1,1,1,1])
# for f in range(0,(len(testX))):
#     flage = 0
#     for i in range(b,cc):
#         # import pdb; pdb.set_trace()
#         if preds[i]==refs[f]:
#             flage = 1
            
#     if flage == 1:
#          end_list_similar.append(list_query[f])
#     if flage !=1:
#         end_list_similar.append(list_similar[b])
#     b = cc
#     cc = (f+2)*maxResults



