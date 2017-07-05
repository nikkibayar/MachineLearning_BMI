"""Purpose of this script is to accurately determine a persons' overall BMI using an image of their face"""

import dlib
import cv2
import numpy
import os
import numpy as np
from sklearn import svm
import sklearn
import re
from sklearn.svm import SVR
import math
import traceback
import time
import shutil
import matplotlib.pyplot as plt


PREDICTOR_PATH = "/Users/nikki-genlife/Desktop/Applications/containers/face/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path = "/Users/nikki-genlife/Desktop/Applications/containers/face/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)
FACE_POINTS = list(range(17, 68))


def calc_BMI(height, weight):
    """
    Calculate BMI using passed in parameters of height & weight.

    Args:
        Param 1 (float): height
        Param 2 (float): weight

    Returns:
        (float) BMI
    """
    h = height / 100
    return weight / (h * h)


def get_feature_ranges(features):
    """
    Finds minimum & maximum values of features in order to be used for normalization

    Args:
        Param 1 (array of integers): Feature vectors

    Returns:
        Param 1 (int): Minimum values of feature vector
        Param 2 (int): Maximum value of feature vector
    """
    component_min = features.min(axis=0)
    component_max = features.max(axis=0)
    return component_min, component_max


def normalize_data(minimum, maximum, data):
    """
    Method takes in feature vectors and normalizes them to ensure every feature has the same weight

    and the classifier does not favor certain features over other ones
    Args:
        Param 1 (int): Minimum value of feature vectors
        Param 2 (int): Maximum valye of feature vectors
        Param 3 (array of integers): feature vectors to normalize

    Returns:
        Normalizes array of feature vectors
    """
    new_min = -1.0
    new_max = 1.0
    for x in range(data.shape[1]):
        data[:, x] = (new_max - new_min) * (data[:, x] - minimum[x]) / \
            (maximum[x] - minimum[x]) + new_min
    return data


def get_landmarks(im):
    """
    Method uses dlib & haarcascades predictors to accurately detect faces and preidict location of facial features.

    Once landmarks are detected they are put into a numpy array to later be used for manipulation & as inputs for the classifier

    Args:
        Param 1: image

    Returns:
        Numpy array of the feature vector of the image passed in OR

        String "bad features" if no landmarks could be detected

    """
    try:
        rects = cascade.detectMultiScale(im, 1.1, 3)
        x = int(rects[0][0])
        y = int(rects[0][1])
        w = int(rects[0][2])
        h = int(rects[0][3])
        rect = dlib.rectangle(x, y, x + w, y + h)
        shape = predictor(im, rect)
        landmarks = numpy.array([[p.x, p.y] for p in shape.parts()])
    except Exception as ex:
        landmarks = "bad features"
    return landmarks


def getWeight(name):
    """
    Decodes weight of image from image name

    Args:
        Param 1 (str): name of image

    Returns:
        Weight OR None if weight could not be found
    """
    i = re.search("W-[0-9]{1,3}.[0-9]{0,3}", name)
    if (i != None):
        weight = i.group(0)[2:]
        return weight
    else:
        return None


def getHeight(name):
    """
    Decodes height of image from image name

    Args:
        Param 1 (str): name of image

    Returns:
        Height OR None if height could not be found
    """
    i = re.search("H-[0-9]{1,3}.[0-9]{0,3}", name)
    if (i != None):
        height = i.group(0)[2:]
        return height
    else:
        return None


def train_bmi_predictor():
    """
    Main method that is used to train classifier. Calls local repo of training images

    and runs landmark detection of them, then normalizes features and uses it, and it's corresponding

    BMI value to train SVR

    Returns:
        Param 1: Global Min value to use for normalization
        Param 2: Global Max value to use for normalization
        Param 3: Training BMI classifier to be used for predicting


    """
    base_dir = '/Users/nikki-genlife/Desktop/anaconda/train_mugshots'
    childrenImages = os.listdir(base_dir)
    landmarks = []
    bmi = []
    for image in range(0, len(childrenImages)):
        pathNameToImage = base_dir + '/' + childrenImages[image]
        features = get_landmarks(cv2.imread(pathNameToImage))
        if(type(features) != str):
            height = getHeight(childrenImages[image])
            weight = getWeight(childrenImages[image])
            if(height != None and weight != None):
                bmi.append(calc_BMI(float(height), float(weight)))
                landmarks.append(features.flatten())
    try:

        landmarks = numpy.array(landmarks)
        Min, Max = get_feature_ranges(landmarks)
        landmarks = np.float32(landmarks)
        landmarks = normalize_data(Min, Max, landmarks)
        print("number of training images was:")
        return Min, Max, bmi_classifier.fit(landmarks, bmi)
    except Exception as ex:
        print(ex)
        pass


def makePredictionForBMI(img):
    """
    Method takes in image and predicts BMI of image after extracting facial landmarks from image

    Args:
        Param 1: image

    Returns:
        Predicted BMI
    """
    try:
        features = get_landmarks(img)

        if(type(features) != str):
            input_landmarks = numpy.array(features.flatten())
            input_landmarks = input_landmarks.reshape(
                1, input_landmarks.shape[0])
            input_landmarks = np.float32(input_landmarks)
            input_landmarks = normalize_data(Min, Max, input_landmarks)
            return bmi_classifier.predict(input_landmarks)
        else:
            return "Couldn't get BMI"
    except ValueError:
        return "Couldn't predict bmi"
        pass


def measure_accuracy():
    """
    Method is used to test accuracy of classifier. Can change base_dir variable to whatever path the images 

    you wish to test on are in

    Returns:
        Nothing - calls calculate_metrics w/ 2 arrays. One array is of the absolute value of true BMI - predicted BMI

        and second array is of all the predicted BMIs
    """
    bmi_difference = []
    all_pred_bmi = []
    base_dir = '/Users/nikki-genlife/Desktop/anaconda/test_mugshots'
    childrenImages = os.listdir(base_dir)
    landmarks = []
    all_data = []
    for image in range(0, len(childrenImages)):
        pathNameToImage = base_dir + '/' + childrenImages[image]
        true_weight = getWeight(childrenImages[image])
        true_height = getHeight(childrenImages[image])
        if(true_height != None and true_weight != None):
            true_bmi = calc_BMI(float(true_height), float(true_weight))
            pred_bmi = makePredictionForBMI(cv2.imread(pathNameToImage))
            if(type(pred_bmi) != str):
                if(abs(true_bmi - pred_bmi) > 7):
                    data = {'image': childrenImages[
                        image], 'error': abs(true_bmi - pred_bmi)}
                    all_data.append(data)
                all_pred_bmi.append(pred_bmi)
                bmi_difference.append(abs(true_bmi - pred_bmi))
    print("Number of testing images was:", len(bmi_difference))
    calculate_metrics(bmi_difference, all_pred_bmi)


def calculate_metrics(bmi_difference, all_predictions):
    """
    Method is sued to calculate all necessary metrics to determine accuracy

    of predictions from classifier

    Args:
        Param 1 (array of floats): bmi differences
        Param 2 (array of floats): all bmi predictions

    Prints:
        Calculated metrics on input arrays
        
    """
    print("Average absolute BMI prediction error:")
    count = 0
    for x in range(0, len(bmi_difference)):
        count = count + bmi_difference[x]
    average = count / len(bmi_difference)
    print(average)
    print("Standard deviation of prediction error:")
    squared = 0
    for x in range(0, len(bmi_difference)):
        squared = squared + (bmi_difference[x] - average) ** 2
    print(math.sqrt(squared / len(bmi_difference)))
    print("Average BMI prediction:")
    count = 0
    for x in range(0, len(all_predictions)):
        count = count + all_predictions[x]
    average = count / len(all_predictions)
    print(average)
    print("Standard deviation of average BMI prediction:")
    squared = 0
    for x in range(0, len(all_predictions)):
        squared = squared + (all_predictions[x] - average) ** 2
    print(math.sqrt(squared / len(all_predictions)))
    print("Time script took to run:")


if __name__ == "__main__":
    Min = 0
    Max = 0
    bmi_model = None
    bmi_classifier = SVR(C=5, gamma=.3)
    start = time.time()
    Min, Max, classifier = train_bmi_predictor()
    measure_accuracy()
    end = time.time()
    print(end - start)
