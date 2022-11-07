####################################################
#Imports
####################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow
import math
import random
import pandas as pd
from PIL import Image

####################################################
#Utilities
####################################################

# Hide and seek pre-processing

def random_hide(img, patch_list, hide_prob=0.5, mean=0.5):
    if type(img) is not np.ndarray:
        img = np.array(img)
    img = img.copy()
    np.random.seed()
    for patch in patch_list:
        (x, y, width, height) = patch
        if np.random.uniform() < hide_prob:
            img[x:x+width, y:y+height] = mean
    return img

def get_random_patch_list(img_size, patch_size):
    if img_size % patch_size != 0:
        raise Exception("patch_size cannot divide by img_size")
    patch_num = img_size//patch_size
    patch_list = [(x*patch_size, y*patch_size, patch_size, patch_size)
                  for y in range(patch_num)
                  for x in range(patch_num)]
    return patch_list

def image_histogram_equalization(image, number_bins=256):

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape)


    y_pred[y_pred >= .5] = 1
    y_pred[y_pred < .5] = 0

    y_pred_mClass = np.zeros([y_pred.shape[0], 1])

    for i in range(0, y_pred.shape[0]):

        if y_pred[i,0] == 0 and y_pred[i,1] == 0:
            y_pred_mClass[i] = 0
        if y_pred[i,0] == 1 and y_pred[i,1] == 0:
            y_pred_mClass[i] = 1
        if y_pred[i,0] == 0 and y_pred[i,1] == 1:
            y_pred_mClass[i] = 2
        if y_pred[i,0] == 1 and y_pred[i,1] == 1:
            y_pred_mClass[i] = 3

    return y_pred_mClass
	
def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

    y_true = np.array(df['group_numeric'])
    y_pred_probabilities = model.predict_generator(data_generator, steps=math.ceil(df.shape[0] / batch_size),
                                                   verbose=1)
    if mode == 'mClass':
        y_pred = np.argmax(y_pred_probabilities, axis=1)
        # Save predictions in excel
        save_excel_predictions_images_name(experiment_folder[0:-1], 'preds_' + name, np.array(df["ImageID"]), y_true,
                                           y_pred_probabilities, classes, 'mLabel')
    if mode == 'mLabel':
        y_pred = recover_mClass_from_mLabel(y_pred_probabilities)
        # Save predictions in excel
        save_excel_predictions_images_name(experiment_folder[0:-1], 'preds_' + name, np.array(df["ImageID"]), y_true,
                                           y_pred_probabilities, ['N', 'I'], mode)

    # Save confusion matrix
    ax = plot_confusion_matrix(y_true, y_pred, np.array(classes))
    ax.figure.savefig(experiment_folder + name)
    plt.close()

    return 1
	

	
