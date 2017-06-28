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

# height_model = None
# weight_model = None
# weight_classifier = SVR()
# height_classifier = SVR()


bmi_model = None
bmi_classifier = SVR()

def calc_BMI(height, weight):
	h = height / 100
	return weight / (h * h)

def split_data():
    all_images = []
    base_dir = '/Users/nikki-genlife/Desktop/anaconda/all_usable_imageNames'
    childrenImages = os.listdir(base_dir)
    for image in range(1, len(childrenImages)):
        pathNameToImage = base_dir + '/' + childrenImages[image]
        all_images.append(pathNameToImage)
   
    random.shuffle(all_images)
    test = all_images[:895]
    train = all_images[895:]
    return test, train

#returns 3 arrays: images, heights, weights
def findUsableImages(images_list):
    images = []
    # heights = []
    # weights = []

    bmi = []
    for image in images_list:
        if (type(get_landmarks(cv2.imread(image))) != str):
            images.append(image)
            # weights.append(getWeight(image))
            # heights.append(getHeight(image))
            w = int(getWeight(image))
            h = int(getHeight(image))
            bmi.append(calc_BMI(h, w))
        else:
            pass
    return images, bmi
    #return images, heights, weights


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
# def train_weight_predictor(image_list, weight_list):
#     try:
#         landmarks = []
#         for image in image_list:
#             features = get_landmarks(cv2.imread(image))
#             landmarks.append(features.flatten());
#         try:
#             landmarks = numpy.array(landmarks)
#             return weight_classifier.fit(landmarks, weight_list)
#         except Exception as ex:
#             print(ex)
#             pass
#     except ValueError:
        #pass

#Trains height predictor
# def train_height_predictor(image_list, height_list):
#     try:
#         landmarks = []
#         for image in image_list:
#             features = get_landmarks(cv2.imread(image))
#             landmarks.append(features.flatten());
#         try:
#             landmarks = numpy.array(landmarks)
#             return height_classifier.fit(landmarks, height_list)
#         except Exception as ex:
#             print(ex)
#             pass
#     except ValueError:
#         pass

#Trains weight predictor
def train_bmi_predictor(image_list, bmi_list):
    try:
        landmarks = []
        for image in image_list:
            features = get_landmarks(cv2.imread(image))
            landmarks.append(features.flatten());
        try:
            landmarks = numpy.array(landmarks)
            return bmi_classifier.fit(landmarks, bmi_list)
        except Exception as ex:
            print(ex)
            pass
    except ValueError:
        pass

#Makes bmi predictions given an image
def makePredictionForBMI(img):
    try:
        features = get_landmarks(img)
        input_landmarks = numpy.array([get_landmarks(img).flatten()])
        return bmi_classifier.predict(input_landmarks)
    except ValueError:
        print("Couldn't predict weight")
        pass

#Makes weight predictions given an image
# def makePredictionForWeight(img):
#     try:
#         features = get_landmarks(img)
#         input_landmarks = numpy.array([get_landmarks(img).flatten()])
#         return weight_classifier.predict(input_landmarks)
#     except ValueError:
#         print("Couldn't predict weight")
#         pass

# #Makes height predictions given an immage
# def makePredictionForHeight(img):
#     try: 
#         features = get_landmarks(img)
#         input_landmarks = numpy.array([get_landmarks(img).flatten()])
#         return height_classifier.predict(input_landmarks)
#     except Exception as ex:
#         print("Couldn't predict height")
#         pass

#Method to serialize classifiers
def pickle_classifiers(weight_model, height_model):
    w = jsonpickle.encode(weight_classifier)
    with open("new_weight_pred", "w") as text_file:
        text_file.write(w)
    
    h = jsonpickle.encode(height_classifier)
    with open("new_height_pred", "w") as text_file:
        text_file.write(h)

split_data()
test_set, train_set = split_data()
usable_test_images, usable_test_bmi = findUsableImages(test_set)
usable_train_images, usable_train_bmi = findUsableImages(train_set)

train_bmi_predictor(usable_train_images, usable_train_bmi)

# test_set, train_set = split_data()
# usable_test_images, usable_test_heights, usable_test_weights = findUsableImages(test_set)
# usable_train_images, usable_train_heights, usable_train_weights = findUsableImages(train_set)
# weight_classifier = train_weight_predictor(usable_train_images, usable_train_weights)
# height_classifier = train_height_predictor(usable_train_images, usable_train_heights)



#Method to measure accuracy
def measure_accuracy(test_data):
    print("in method")
    bmi_difference = []
    all_pred_bmi = []
    for image in test_data:
        true_weight = int(getWeight(image))
        true_height = int(getHeight(image))
        true_bmi = calc_BMI(true_height, true_weight)
        if (type(get_landmarks(cv2.imread(image))) != str):
            if(type(get_landmarks(cv2.imread(image))) != str):
                pred_bmi = makePredictionForBMI(cv2.imread(image))
                all_pred_bmi.append(pred_bmi)
                bmi_difference.append(abs(true_bmi - pred_bmi))
    

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
print("here ")
measure_accuracy(usable_test_images)

# #Method to measure accuracy
# def measure_accuracy(test_data):
#     bmi_difference = []
#     all_pred_bmi = []
#     for image in test_data:
#         true_weight = int(getWeight(image))
#         true_height = int(getHeight(image))
#         true_bmi = calc_BMI(true_height, true_weight)
#         if (type(get_landmarks(cv2.imread(image))) != str):
#             if(type(get_landmarks(cv2.imread(image))) != str):
#                 pred_weight = float(makePredictionForWeight(cv2.imread(image)))
#                 pred_height = float(makePredictionForHeight(cv2.imread(image)))
#                 pred_bmi = calc_BMI(pred_height, pred_weight)
#                 all_pred_bmi.append(pred_bmi)
#                 bmi_difference.append(abs(true_bmi - pred_bmi))
    

#     sum = 0
#     for num in bmi_difference:
#         sum = sum + num
#     print("BMI predictions were off by an average of:")
#     print(sum / len(bmi_difference))
  
#     sum = 0
#     for num in all_pred_bmi:
#         sum = sum + num
#     print("BMI predictions are on average: ")
#     print(sum / len(all_pred_bmi))

# measure_accuracy(test_set)



















