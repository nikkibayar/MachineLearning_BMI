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


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def extract_good_features(l):
    features = []
    #Ratio of cheekbone width to jaw width
    feat_1 = (l[15][0] - l[1][0] ) / (l[12][0] - l[4][0])
    features.append(feat_1)

    
    #Ratio of beeckbone width to upper facial height
    feat_2 = (l[12][0] - l[4][0]) / (l[27][1] * l[66][1])
    features.append(feat_2)

    #Ratio of the permiter area of polygon running through lower face to area of polygon 
    side_1 = l[15][0] - l[1][0]
    side_2 = math.sqrt(((l[15][1] - l[12][1]) **2) + (l[15][0] - l[12][0]) **2)
    side_3 = math.sqrt(((l[12][0] - l[8][0]) **2) + (l[12][1] - l[8][1]) **2)
    side_4 = math.sqrt(((l[8][0] - l[4][0]) **2) + (l[4][1] - l[8][1]) **2)
    side_5 = math.sqrt(((l[4][0] - l[1][0]) **2) + (l[1][1] - l[4][1]) **2)
    
    tri_1 = .5 * ((l[4][0] - l[1][0])  * (l[1][1] - l[4][1]))
    tri_2 = .5 * ((l[15][0] - l[12][0])  * (l[15][1] - l[12][1]))
    tri_3 = .5 * ((l[12][0] - l[4][0])  * (l[12][1] - l[8][1]))
    square = ((l[12][0] - l[4][0])  * (l[15][1] - l[12][1]))
    feat_3 = (side_5 + side_4 + side_3 + side_2 + side_1) / (tri_3 + tri_2 + tri_1 + square) 
    features.append(feat_3)

    
    #Average side of eyes 
    feat_4 = .5 * ((l[45][0] - l[36][0]) - (l[42][0] - l[39][0]))
    features.append(feat_4)
    
    
    #Ratio of lower face to face height
    line_1 = line(l[36], l[19])
    line_2 = line(l[24], l[45])
    R = intersection(line_1, line_2)
    if (R != False):
        feat_5 = (l[15][1] - l[8][1]) / (R[1] - l[8][1])
        features.append(feat_5)
    
    #Ratio of face width to lower fasce height 
        feat_6 = (l[15][0] - l[1][0]) / (l[15][1] - l[8][1])
        features.append(feat_6)


        #Average distance between eyebrows & upper edge of eyes
        feat_7 = (1/6) * ((l[17][1]  - l[36][1]) + (l[19][1] - l[37][1]) + (l[21][1] - 
            l[39][1]) + (l[22][1] - l[42][1]) + (l[24][1] - l[44][1]) + (l[26][1] - l[45][1]))
        features.append(feat_7)
    #print(features)
        return features
    return "Bad features"
    


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
        landmarks = "Couldn't get features"
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
    i = re.search("-[0-9]{1,3}.[0-9]{0,3}", name)
    if (i != None):
        height = i.group(0)[1:]
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
    base_dir = '/Users/nikki-genlife/Desktop/anaconda/Tr_mugshots'
    childrenImages = os.listdir(base_dir)
    landmarks = []
    bmi = []
    for image in range(0, len(childrenImages)):
        pathNameToImage = base_dir + '/' + childrenImages[image]
        features = get_landmarks(cv2.imread(pathNameToImage))
        if (len(features) == 68):
            features = extract_good_features(features)
            if(type(features) != str):
                height = getHeight(childrenImages[image])
                weight = getWeight(childrenImages[image])
                if(height != None and weight != None):
                    bmi.append(calc_BMI(float(height), float(weight)))
                    landmarks.append(features)
    try:
        landmarks = numpy.array(landmarks)
        # landmarks, averages, SDs = normalize_data(landmarks)
        print("number of training images was:", len(landmarks))
        #return Min, Max, bmi_classifier.fit(landmarks, bmi)
        return bmi_classifier.fit(landmarks, bmi)
    except Exception as ex:
        print(ex)
        pass

# def norm(landmarks):
#     print(type(landmarks))
#     print(landmarks)
#     print(averages)
#     print(SDs)
#     print(landmarks[4])
#     for x in range(0, len(landmarks)):
#         landmarks[x] = (landmarks[x] - averages[x]) / SDs[x]
#     return landmarks


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
            # input_landmarks = numpy.array(features)
            input_landmarks = extract_good_features(features)
            print("bmi predictin is:", bmi_classifier.predict(input_landmarks))
            return bmi_classifier.predict(input_landmarks)
        else:
            return "Couldn't get BMI"
    except ValueError:
        return "Couldn't predict bmi"


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
    base_dir = '/Users/nikki-genlife/Desktop/anaconda/T_mugshots'
    childrenImages = os.listdir(base_dir)
    landmarks = []
    for image in range(0, len(childrenImages)):
        pathNameToImage = base_dir + '/' + childrenImages[image]
        true_weight = getWeight(childrenImages[image])
        true_height = getHeight(childrenImages[image])
        if(true_height != None and true_weight != None):
            true_bmi = calc_BMI(float(true_height), float(true_weight))
            pred_bmi = makePredictionForBMI(cv2.imread(pathNameToImage))
            if(type(pred_bmi) != str):
                all_pred_bmi.append(pred_bmi)
                bmi_difference.append(abs(true_bmi - pred_bmi))
    print("Number of testing images was:", len(bmi_difference))
    calculate_metrics(bmi_difference, all_pred_bmi)


def calculate_metrics(bmi_difference, all_predictions):
    """
    Method is used to calculate all necessary metrics to determine accuracy

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
    pca = 0
    averages = 0
    SDs = 0
    bmi_classifier = SVR(C=4.5, gamma=1/float(200), epsilon=.1)
    start = time.time()
    bmi_classifier = train_bmi_predictor()
    print(bmi_classifier)
    measure_accuracy()
    end = time.time()
    print(end - start)
