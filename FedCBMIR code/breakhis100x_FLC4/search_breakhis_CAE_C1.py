##
#GPU

# 
# from numba import cuda 
# device = cuda.get_current_device() 
# device.reset()

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


#Directions
args1 = {
	# "index": "C:/Users/Zahra/Downloads/archive/BreaKHis_400X/index128.csv" # extracted feature from training data
    "index": "/home/yuandou/test/CBMIR_breastcancer_FLC1/check/SE/index128.csv"
}

# load our autoencoder from disk
run_folders1 = {
    # "weights_path":"C:/Users/Zahra/Downloads/archive/BreaKHis_400X/exp_0008/weights/VAE_weights.h5",
    # "tsv_path_test": "C:/Users/Zahra/Downloads/archive/BreaKHis_400X/test/test/test.csv",
    # "tsv_path_train":"C:/Users/Zahra/Downloads/archive/BreaKHis_400X/train/train/train.csv",

    "weights_path":"/home/yuandou/test/CBMIR_breastcancer_FLC1/check/FL.CAE.VAE.beforeTSNE/exp_0004/round_0002/weights/VAE_weights.h5",
    "tsv_path_test": "/home/yuandou/test/BreaKHis 400X/test/test/test.csv",
    "tsv_path_train": "/home/yuandou/test/BreaKHis 400X/train/train/x_train_Break_FDL_CL1.csv",

}

#%%Reading Data
batch_size = 1
backbone = 'fsconv'
classes = ['label']
input_shape = (128, 128, 3)
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

trainX = data_generator_train.flow_from_dataframe(dataframe=pd.read_csv('/home/yuandou/test/BreaKHis 400X/train/train/x_train_Break_FDL_CL1.csv'),    # to change path
                                                directory='/home/yuandou/test/BreaKHis 400X/images',   # to change path
                                                x_col='Image_name',
                                                y_col=classes,
                                                batch_size=batch_size,
                                                seed=42,
                                                shuffle=True,
                                                class_mode='raw',
                                                target_size=input_shape[:-1])

data_generator_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
testX = data_generator_test.flow_from_dataframe(dataframe=pd.read_csv('/home/yuandou/test/BreaKHis 400X/test/test/test.csv'),     # path change
                                               directory='/home/yuandou/test/BreaKHis 400X/images',      # path change
                                               x_col='Image_name',
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
        with open('/home/yuandou/test/CBMIR_breastcancer_FLC1/check/CL.CAE.VAE.beforeTSNE/exp_0003/round_0001/VAE_model.pkl', 'rb') as f:  # add final model path
            params = pickle.load(f)
        my_VAE = ConvVarAutoencoder(*params)
        my_VAE.build(use_batch_norm=True, use_dropout=True)
        my_VAE.model.load_weights(os.path.join(os.path.join(run_folders1["weights_path"])))
        return my_VAE

autoencoder = load_model_AD1("VAE_model.pkl")
print(autoencoder.model.summary())
#%%
FE = Model(inputs=autoencoder.model.get_layer("encoder_input").input,
 	outputs=autoencoder.model.get_layer("encoder_output").output)
print(FE.summary())
# Learning curve, confusion matrix and predictions
features = FE.predict(trainX, math.ceil(trainX.n / batch_size))
# features = features[:,:48]
# image to its corresponding latent-space representation
indexes = list(range(0, len(trainX)))
data = {"indexes": indexes, "features": features}
# write the data dictionary to disk
# print("[INFO] saving index...")
# f = open(args1["index"], "wb")
# f.write(pickle.dumps(data))
# f.close()

######################
#%%CON2

from scipy.spatial import distance
from scipy.spatial.distance import cityblock, cosine
def euclidean(a, b):
	euclidean = np.linalg.norm(a - b) #cosine_similarity = 1 - cosine(a, b)
    #distance1= distance.hamming(a,b) #euclidean = np.linalg.norm(a - b) #manhattan=distance.cityblock(a,b) # compute and return the euclidean distance between two vectors
	return euclidean

def perform_search(queryFeatures, index, maxResults=1):
	# initialize our list of results
	results = []
	# loop over our index
	for i in range(0, len(index["features"])):
		# compute the euclidean distance between our query features
		# and the features for the current image in our index, then
		# update our results list with a 2-tuple consisting of the
		# computed distance and the index of the image
		d = euclidean(queryFeatures, index["features"][i])
		results.append((d, i)); 
        # import pdb; pdb.set_trace()
	# sort the results and grab the top ones
	results = sorted(results)[ : maxResults]
	# return the list of results
	return results; 
# construct the argument parse and parse the arguments
# results = sorted(results)[ : maxResults]

#%%
# tuning the number of results and query images
maxResults = 3
# size = 25
# loading indexs
from sklearn.metrics import top_k_accuracy_score
# index = pickle.loads(open(args1["index"], "rb").read())
index = data
features_test = FE.predict(testX, math.ceil(testX.n / batch_size))
print("HERE")
#%%

# features_test = features_test[:,:48]
#%%
list_similar = []
testlist = []
query_name = []
images1 = [];
list_query = []
queryIdxs = [ 0, 1]
#searching and finding results      
for i in range(0, len(testX)):
    queryFeatures = features_test[i]
    results = perform_search(queryFeatures, index , maxResults)    
    #loop over the Results    
    for (d , j) in results:
        # import pdb; pdb.set_trace()
        list_similar.append(trainX.labels[j])
    list_query.append(testX.labels[i])
    #similar images
#evaluate
# Reset data generator after prediction and set references
preds = list_similar  # Predictions (Label)
refs = list_query  # References
end_list_similar = []

from sklearn.metrics import accuracy_score
#%%
end_list_similar = []
flage = 0
b = 0
i = 0
f = 0
cc = maxResults
pre = []
yesyes = []
# nothing = np.array([1,1,1,1])
for f in range(0,(len(testX))):
    flage = 0
    yes = 0
    for i in range(b,cc):
        # import pdb; pdb.set_trace()
        if preds[i]==refs[f]:
            flage = 1
            yes += 1 
    if flage == 1:
         end_list_similar.append(list_query[f])
    if flage !=1:
        end_list_similar.append(list_similar[b]) #if flag != 1, write the toppest
    b = cc
    cc = (f+2)*maxResults
    norm_yes = yes/maxResults
    pre.append(norm_yes)
#%%##########
# inja dg nmishe az argmax estefade kard.   dar in halat acc ra ba hamun 4 sotun mohasebe mikonim.
      
preds_top = end_list_similar  # Predictions (Label)
refs = list_query  # References
# print("preds_top = " + str(preds_top))
# print("refs = " + str(refs))
# print("end_list_similar = " + str(end_list_similar))

from sklearn.metrics import accuracy_score
accuracy1 = accuracy_score(list_query, end_list_similar)
print ("accuracy1 is = " + str(accuracy1))

cm_search1 = []
labels1= ["NC", "G3"]
cm_search1 = confusion_matrix(refs, preds_top, labels= [0, 1])
print("cm_search")
print(cm_search1)
cm = ConfusionMatrixDisplay(cm_search1, display_labels= labels1)
cm.plot()
cm.ax_.set(title = "k = 3, Euclidean")
#####
class_names = [0, 1]
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
disp = ConfusionMatrixDisplay(confusion_matrix = cm_search1,
                              display_labels = [0, 1])

disp = disp.plot()
plt.show()
#%% precision_score
import sklearn
precision = sklearn.metrics.precision_score(refs, preds_top)
print("precision = " + str (precision))
recall = sklearn.metrics.recall_score(refs, preds_top)
print("recall = " + str (recall))
accuracy1 = accuracy_score(refs, preds_top)
print ("accuracy1 is = " + str(accuracy1))
from sklearn.metrics import f1_score
f1_score = f1_score(refs, preds_top)
print ("f1_score is = " + str(f1_score))
recall = sklearn.metrics.recall_score(refs, preds_top)
print("recall = " + str (recall))
######
#Report
# from sklearn.metrics import classification_report
# target_names = ['NC','t']
# print(classification_report(refs, preds_top, target_names=target_names))
#%% plotting features

# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2).fit_transform(features[:])
# #%%
# df = pd.DataFrame()
# y_train = (trainX.labels[:])
# df["comp-1"] = tsne[:,0]
# df["comp-2"] = tsne[:,1]
# df["y"] = y_train
# import seaborn as sns
# sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
#                 palette=sns.color_palette("hls", 2),
#                 data=df).set(title="T-SNE projection")
# #%%
# sum_pre = sum(pre)
# avg = sum_pre/2122
# print(avg)