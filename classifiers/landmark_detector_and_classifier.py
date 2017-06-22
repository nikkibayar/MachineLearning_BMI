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
import jsonpickle
from numpy import ndarray
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

height_model = None
weight_model = None
weight_classifier = SVR()
height_classifier = SVR()

def split_data():
    all_images = []
    base_dir = '/Users/nikki-genlife/Desktop/anaconda/all_usable_imageNames'
    childrenImages = os.listdir(base_dir)
    for image in range(1, len(childrenImages)):
        pathNameToImage = base_dir + '/' + childrenImages[image]
        all_images.append(pathNameToImage)
   
    random.shuffle(all_images)
    test = all_images[:520]
    train = all_images[520:]
    return test, train

#returns 3 arrays: images, heights, weights
def findUsableImages(images_list):
    images = []
    heights = []
    weights = []
    for image in images_list:
        if (type(get_landmarks(cv2.imread(image))) != str):
            images.append(image)
            weights.append(getWeight(image))
            heights.append(getHeight(image))
        else:
            pass
    return images, heights, weights


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
    i = re.search("W-[0-9]{1,3}", name)
    weight = i.group(0)[2:]
    return weight

#Finds height in a given line
def getHeight(name):
    i = re.search("H-[0-9]{1,3}", name)
    height = i.group(0)[2:]
    return height
    
#Trains weight predictor
def train_weight_predictor(image_list, weight_list):
    try:
        landmarks = []
        for image in image_list:
            features = get_landmarks(cv2.imread(image))
            landmarks.append(features.flatten());
        try:
            landmarks = numpy.array(landmarks)
            return weight_classifier.fit(landmarks, weight_list)
        except Exception as ex:
            print(ex)
            pass
    except ValueError:
        pass

#Trains height predictor
def train_height_predictor(image_list, height_list):
    try:
        landmarks = []
        for image in image_list:
            features = get_landmarks(cv2.imread(image))
            landmarks.append(features.flatten());
        try:
            landmarks = numpy.array(landmarks)
            return height_classifier.fit(landmarks, height_list)
        except Exception as ex:
            print(ex)
            pass
    except ValueError:
        pass

#Makes weight predictions given an image
def makePredictionForWeight(img):
    try:
        features = get_landmarks(img)
        input_landmarks = numpy.array([get_landmarks(img).flatten()])
        print("predicted weight is:")
        return weight_classifier.predict(input_landmarks)
    except ValueError:
        print("Couldn't predict weight")
        pass

#Makes height predictions given an immage
def makePredictionForHeight(img):
    try: 
        features = get_landmarks(img)
        input_landmarks = numpy.array([get_landmarks(img).flatten()])
        print("prediction height is:")
        return height_classifier.predict(input_landmarks)
    except Exception as ex:
        print("Couldn't predict height")
        pass

#Method to serialize classifiers
def pickle_classifiers(weight_model, height_model):
    w = jsonpickle.encode(weight_classifier)
    with open("new_weight_pred", "w") as text_file:
        text_file.write(w)
    
    h = jsonpickle.encode(height_classifier)
    with open("new_height_pred", "w") as text_file:
        text_file.write(h)


test_set, train_set = split_data()
usable_test_images, usable_test_heights, usable_test_weights = findUsableImages(test_set)
usable_train_images, usable_train_heights, usable_train_weights = findUsableImages(train_set)
weight_classifier = train_weight_predictor(usable_train_images, usable_train_weights)
height_classifier = train_height_predictor(usable_train_images, usable_train_heights)

# img = cv2.imread('/Users/nikki-genlife/Desktop/mason.jpg')
# makePredictionForHeight(img)
# makePredictionForWeight(img)


def measure_accuracy(test_data):
    height_difference = []
    weight_difference = []
    for image in test_data:
        true_weight = int(getWeight(image))
        print(true_weight)
        true_height = int(getHeight(image))
        print(true_height)
        if (type(get_landmarks(cv2.imread(image))) != str):
            if(type(get_landmarks(cv2.imread(image))) != str):
                pred_weight = float(makePredictionForWeight(cv2.imread(image)))
                pred_height = float(makePredictionForHeight(cv2.imread(image)))
                weight_difference.append(abs(true_weight - pred_weight))
                height_difference.append(abs(true_height - pred_height))
        
    sum = 0
    for num in height_difference:
        sum = sum + num
    print("height predicions were off (in cm) by an average of:")
    print(sum / len(height_difference))
    sum = 0
    for num in weight_difference:
        sum = sum + num
    print("weight predicions were off (in kg) by an average of:")
    print(sum / len(weight_difference))

measure_accuracy(train_set)





















