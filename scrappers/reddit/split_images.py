import numpy as np
import cv2
import os
import math
import json
import copy

MAX_SIDE = 1000
ROOT_FOLDER = "images_filtered"
DEBUG_PATH = "images_debug"
FINAL_FOLDER = "images_dataset"
BASE_JSON = "filtered_data.json"
GROUDTRUTH_JSON = "groundtruth.json"

def get_lines(raw_image):
    image_gray = cv2.cvtColor(raw_image,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image_gray,50,150,apertureSize = 3)

    lines = []

    lines_temp = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines_temp is None:
        return []

    for rho,theta in lines_temp[0]:
        if math.degrees(theta) != 0.0:
            continue


        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 10000*(-b))
        y1 = int(y0 + 10000*(a))
        x2 = int(x0 - 10000*(-b))
        y2 = int(y0 - 10000*(a))
        lines.append((x1,y1,x2,y2))

    return lines

def get_central_line(lines, shape):
    if len(lines) == 1:
        return lines[0]

    _, width,_ = shape
    if len(lines) > 1:
        lines_x = [ np.abs(x - width/2.0) for (x,_,_,_) in lines ]
        return lines[ lines_x.index( min(lines_x) ) ]

def draw_line(image_path):
    raw_image = cv2.imread(image_path)
    lines = get_lines(raw_image)

    if len(lines) > 0:
        central_line = get_central_line(lines, raw_image.shape)
        cv2.line(raw_image,(central_line[0], central_line[1]),(central_line[2], central_line[3]),(0,255,0),15)
        cv2.line(raw_image,(central_line[0], central_line[1]),(central_line[2], central_line[3]),(0,0,255),5)
    else:
        return False

    cv2.imwrite( image_path.replace(ROOT_FOLDER, DEBUG_PATH), raw_image )
    return True

def get_two_images(path):
    raw_image = cv2.imread(path)
    lines = get_lines(raw_image)
    x,_,_,_ = get_central_line(lines, raw_image.shape)

    return ( raw_image[:, 0:x ], raw_image[:, x: ] )


def split_image_in_two():
    images_to_be_split = [filename for filename in os.listdir(DEBUG_PATH)]
    for image in images_to_be_split:
        left_image, right_image = get_two_images(os.path.join(ROOT_FOLDER, image))
        filename, ext = os.path.splitext(image)
        cv2.imwrite( os.path.join(FINAL_FOLDER, '{}_left{}'.format(filename, ext) ), left_image )
        cv2.imwrite( os.path.join(FINAL_FOLDER, '{}_right{}'.format(filename, ext) ), right_image )

def generate_debug_lines():
    error_images= []
    splits_not_ok = []
    for filename in os.listdir(ROOT_FOLDER):
        print(filename)
        try:
            split_ok = draw_line(os.path.join(ROOT_FOLDER, filename))
            if split_ok is False:
                splits_not_ok.append(filename)

        except:
            print("Error")
            error_images.append(filename)

    if len(error_images) > 0:
        print("The following images were not processed due to errors")
        print(error_images)
    else:
        print("No errors while processing the images")

    if len(splits_not_ok) > 0:
        print("Couldnt split the following images")
        print(splits_not_ok)
    else:
        print("All images have been split ok.")

def convert_to_kgs(weight):
    return str( int(float(weight)*0.453592) )

def convert_to_cms(height):
    height = height[:height.find(u'\u201d' )]
    height = height.replace('\'', '.')
    return str(int(float(height)*30.48))

def split_data_in_json():
    with open(BASE_JSON) as data_file:
        base_metadata = json.load(data_file)
        base_metadata = {x['id']: x for x in base_metadata}



    groundtruth = []
    for filename in os.listdir(FINAL_FOLDER):
        filename, _ = os.path.splitext(filename)

        id, position = filename.split("_")
        base_data = base_metadata[id]
        object = copy.deepcopy(base_data)
        object['id'] = filename

        after_weight = object.pop('after_weight')
        before_weight = object.pop('before_weight')
        if position == "right":
            object[u'weight'] = convert_to_kgs(after_weight)
        else:
            object[u'weight'] = convert_to_kgs(before_weight)

        object[u'height'] = convert_to_cms(object[u'height'])

        groundtruth.append(object)

    with open(GROUDTRUTH_JSON , 'w') as outfile:
        json.dump(groundtruth, outfile, indent=4, sort_keys=True)



if __name__ == "__main__":
    # generate_debug_lines()
    # split_image_in_two()
    split_data_in_json()