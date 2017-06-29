import dlib
import cv2
import numpy
import numpy as np
import os
from sklearn import svm
import sklearn
from sklearn.svm import SVR
import glob, random, math, itertools
import re
import pickle
import traceback
import time
import shutil
import json
import jsonpickle
from sklearn.model_selection import GridSearchCV
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()

PREDICTOR_PATH = "/Users/nikki-genlife/Desktop/Applications/containers/face/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path = "/Users/nikki-genlife/Desktop/Applications/containers/face/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))
CHIN_POINTS = list(range(6, 11))

start = time.time()
parameters = {'C':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'gamma':
              [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.2, 0.3, 0.4, 0.5]}

bmi_model = None
bmi_classifier = SVR(C=5, gamma=.3)
#grid = GridSearchCV(bmi_classifier, parameters, n_jobs=4)

#Calculate BMI when given height & weight in cm and kg
def calc_BMI(height, weight):
    h = height / 100
    return weight / (h * h)

#Detects landmarks
def get_landmarks(im):
    try:
        rects = cascade.detectMultiScale(im, 1.1, 3)
        x = int(rects[0][0])
        y = int(rects[0][1])
        w = int(rects[0][2])
        h = int(rects[0][3])
        rect = dlib.rectangle(x, y, x+w, y+h)
        shape = predictor(im, rect)
        landmarks = numpy.array([[p.x, p.y] for p in shape.parts()])
    except Exception as ex:
        landmarks = "bad features"
    return landmarks

#Finds weight in a given line
def getWeight(name):
    i = re.search("W-[0-9]{1,3}.[0-9]{0,3}", name)
    if (i != None):
        weight = i.group(0)[2:]
        return weight
    else:
        return None

#Finds height in a given line
def getHeight(name):
    i = re.search("H-[0-9]{1,3}.[0-9]{0,3}", name)
    if (i != None):
        height = i.group(0)[2:]
        return height
    else:
        return None

#Trains predictor from images inside the training set data
def train_bmi_predictor():
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
        print("number of training images was:")
        print(len(landmarks))
        #grid.fit(landmarks, bmi)
        #print(grid.best_params_)
        return bmi_classifier.fit(landmarks, bmi)
    except Exception as ex:
        print(ex)
        pass

train_bmi_predictor()

#Makes prediction for the image passed in as an argument
def makePredictionForBMI(img):
    try:
        features = get_landmarks(img)
        if(type(features) != str):
            input_landmarks = numpy.array([get_landmarks(img).flatten()])
            return bmi_classifier.predict(input_landmarks)
        else:
            return "Couldn't get BMI"
    except ValueError:
        return "Couldn't predict bmi"
        pass

#Method to serialize classifiers: Not called for now
def pickle_classifiers(weight_model, height_model):
    w = jsonpickle.encode(weight_classifier)
    with open("new_weight_pred", "w") as text_file:
        text_file.write(w)
    h = jsonpickle.encode(height_classifier)
    with open("new_height_pred", "w") as text_file:
        text_file.write(h)

#Method to measure accuracy
def measure_accuracy():
    print("in method")
    bmi_difference = []
    all_pred_bmi = []
    base_dir = '/Users/nikki-genlife/Desktop/anaconda/test_mugshots'
    childrenImages = os.listdir(base_dir)
    landmarks = []
    count = 0
    for image in range(0, len(childrenImages)):
        pathNameToImage = base_dir + '/' + childrenImages[image]
        true_weight = getWeight(childrenImages[image])
        true_height = getHeight(childrenImages[image])
        if(true_height != None and true_weight != None):
            true_bmi = calc_BMI(float(true_height), float(true_weight))
            pred_bmi = makePredictionForBMI(cv2.imread(pathNameToImage))
            if(type(pred_bmi) != str):
                count = count + 1
                all_pred_bmi.append(pred_bmi)
                bmi_difference.append(abs(true_bmi - pred_bmi))
    print("number of testing images was:")
    print(count)
    sum = 0
    for num in bmi_difference:
        sum = sum + num
    print("BMI predictions were off by an average of:")
    print(sum / len(bmi_difference))
    sum = 0
    for num in all_pred_bmi:
        sum = sum + num
    print("BMI predictions are on average: ")
    print(sum / len(all_pred_bmi))
    end = time.time()
    print(end-start)

measure_accuracy()


#SAVING METHODS FOR LATER: USED FOR SPLITTING DATABASE INTO TESTING AND TRAINING SETS   
# def send_to_folder(folder, folder_name):
#     for image in range(0, len(folder)):
#         print(folder[image])
#         old_path = folder[image]['path'] 
#         print(old_path)
#         new_path = '/Users/nikki-genlife/Desktop/anaconda/' + str(folder_name) + '/' + folder[image]['name']
#         print(new_path)
#         os.rename(old_path, new_path)

# #Usable Images
# def split_data():
#     all_images = []
#     base_dir = '/Users/nikki-genlife/Desktop/anaconda/mugshots'
#     childrenImages = os.listdir(base_dir)
#     chilrenNames = []
#     for image in range(1, len(childrenImages)):
#         childrenImages.append(childrenImages[image])
#         pathNameToImage = base_dir + '/' + childrenImages[image]
#         data = {'path': pathNameToImage, 'name': childrenImages[image]}
#         all_images.append(data)
#     random.shuffle(all_images)
#     test = all_images[:471]
#     train = all_images[471:]
#     print("here")
#     send_to_folder(train, 'train_mugshots')
#     send_to_folder(test, 'test_mugshots')
#     return test, train
# test_set, train_set = split_data()













