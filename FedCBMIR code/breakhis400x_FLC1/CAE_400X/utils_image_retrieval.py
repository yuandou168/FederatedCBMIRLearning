import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json

def learning_curve_plot(history, run_folders):

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.axis([0, history.epoch[-1], 0, max(history.history['loss'] + history.history['val_loss'])])
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(run_folders["model_path"] + run_folders["exp_name"]+"/viz/"+"training_loss.png")
    plt.close()


# def save_reconstructed_images(y_true, y_pred, run_folders):
#     for i in range(0, y_pred.shape[0]):
#         original = (y_true[i] * 255).astype("uint8")
#         recon = (y_pred[i] * 255).astype("uint8")
#         output = np.hstack([original, recon])
#         cv2.imwrite(run_folders["results_path"] + run_folders["exp_name"] + "/image_" + str(i) + ".png", output)
def save_reconstructed_images(y_true, y_pred, run_folders):
    for i in range(0, y_pred.shape[0]):
        
        original = (y_true[i] * 255).astype("uint8")
        recon = (y_pred[i] * 255).astype("uint8")
        # print(recon[i])
        print(original[i])
        output = np.hstack([original,recon])
        cv2.imwrite(run_folders["results_path"] + run_folders["exp_name"] + "/image_" + str(i) + ".png", output)



def create_json(hyperparameters, run_folders):
    with open(run_folders["model_path"]+run_folders["exp_name"]+"/hyperparameters.json", 'w') as fp:
        json.dump(hyperparameters, fp)


def create_environment(run_folders):
    # Creating base folders
    try:
        os.mkdir(run_folders["model_path"])
    except:
        pass
    try:
        os.mkdir(run_folders["results_path"])
    except:
        pass

    # Preparing required I/O paths for each experiment
    if len(os.listdir(run_folders["model_path"])) == 0:
        exp_idx = 1
    else:
        exp_idx = len(os.listdir(run_folders["model_path"])) + 1

    exp_name = "exp_%04d" % exp_idx
    run_folders["exp_name"] = exp_name

    exp_model_folder = run_folders["model_path"] + run_folders["exp_name"] + '/'
    exp_res_model = run_folders["results_path"] + run_folders["exp_name"] + '/'

    try:
        os.mkdir(exp_model_folder)
    except:
        pass
    try:
        os.mkdir(exp_res_model)
    except:
        pass
    try:
        os.mkdir(os.path.join(exp_model_folder, 'viz'))
    except:
        pass
    try:
        os.mkdir(os.path.join(exp_model_folder, 'weights'))
    except:
        pass
    try:
        os.mkdir(os.path.join(exp_model_folder, 'images'))
    except:
        pass


def euclidean(a, b):
    return np.linalg.norm(a - b)


def perform_search(queryFeatures, index, maxResults=64):
    results = []
    for i in range(0, len(index["features"])):
        d = euclidean(queryFeatures, index["features"][i])
        results.append((d, i))
        results = sorted(results)[:maxResults]
    return results


def build_montages(image_list, image_shape, montage_shape):
    """
    ---------------------------------------------------------------------------------------------
    author: Kyle Hounslow
    ---------------------------------------------------------------------------------------------
    Converts a list of single images into a list of 'montage' images of specified rows and columns.
    A new montage image is started once rows and columns of montage image is filled.
    Empty space of incomplete montage images are filled with black pixels
    ---------------------------------------------------------------------------------------------
    :param image_list: python list of input images
    :param image_shape: tuple, size each image will be resized to for display (width, height)
    :param montage_shape: tuple, shape of image montage (width, height)
    :return: list of montage images in numpy array format
    ---------------------------------------------------------------------------------------------
    example usage:
    # load single image
    img = cv2.imread('lena.jpg')
    # duplicate image 25 times
    num_imgs = 25
    img_list = []
    for i in xrange(num_imgs):
        img_list.append(img)
    # convert image list into a montage of 256x256 images tiled in a 5x5 montage
    montages = make_montages_of_images(img_list, (256, 256), (5, 5))
    # iterate through montages and display
    for montage in montages:
        cv2.imshow('montage image', montage)
        cv2.waitKey(0)
    ----------------------------------------------------------------------------------------------
    """
    if len(image_shape) != 2:
        raise Exception('image shape must be list or tuple of length 2 (rows, cols)')
    if len(montage_shape) != 2:
        raise Exception('montage shape must be list or tuple of length 2 (rows, cols)')
    image_montages = []
    # start with black canvas to draw images onto
    montage_image = np.zeros(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                          dtype=np.uint8)
    cursor_pos = [0, 0]
    start_new_img = False
    for img in image_list:
        if type(img).__module__ != np.__name__:
            raise Exception('input of type {} is not a valid numpy array'.format(type(img)))
        start_new_img = False
        img = cv2.resize(img, image_shape)
        # draw image to black canvas
        montage_image[cursor_pos[1]:cursor_pos[1] + image_shape[1], cursor_pos[0]:cursor_pos[0] + image_shape[0]] = img
        cursor_pos[0] += image_shape[0]  # increment cursor x position
        if cursor_pos[0] >= montage_shape[0] * image_shape[0]:
            cursor_pos[1] += image_shape[1]  # increment cursor y position
            cursor_pos[0] = 0
            if cursor_pos[1] >= montage_shape[1] * image_shape[1]:
                cursor_pos = [0, 0]
                image_montages.append(montage_image)
                # reset black canvas
                montage_image = np.zeros(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                                      dtype=np.uint8)
                start_new_img = True
    if start_new_img is False:
        image_montages.append(montage_image)  # add unfinished montage
    return image_montages
