from scipy.io import loadmat
from datetime import datetime
import math
import cv2
import os
import json

def load_data(path):
    samples = []
    for dataset in next(os.walk(path))[1]:
        path_to_dataset  = os.path.join(path, dataset)
        with open(os.path.join(path_to_dataset, 'groundtruth.json')) as data_file:
            data = json.load(data_file)

        for x in data:
            id = x.pop('id')
            x['full_path'] = os.path.join(path_to_dataset ,id)
        samples.extend(data)

        print("Read {} entries for dataset: {}".format(len(data), dataset))
    return samples

def calc_age(photo_taken_date, matlab_date):
    birth = datetime.fromordinal(max(int(matlab_date) - 366, 1))

    if birth.month < 7:
        return photo_taken_date - birth.year
    else:
        return photo_taken_date - birth.year - 1

def is_image_good(path):
    if os.path.isfile(path) is False:
        return False

    img = cv2.imread(path)
    if img is None:
        return False

    shape = img.shape
    if (shape[0] < 20) or (shape[1] < 20):
        return False

    return True

def convert_mat_to_json(path, dataset):

    mat_file_name = os.path.join(path, "{}.mat".format(dataset) )
    meta = loadmat(mat_file_name)
    dob = meta[dataset][0, 0]["dob"][0]
    photo_taken = meta[dataset][0, 0]["photo_taken"][0]  # year
    full_path = meta[dataset][0, 0]["full_path"][0]
    gender = meta[dataset][0, 0]["gender"][0]
    face_score = meta[dataset][0, 0]["face_score"][0]
    second_face_score = meta[dataset][0, 0]["second_face_score"][0]
    face_location = meta[dataset][0, 0]["face_location"][0]

    total_samples = len(photo_taken)

    #adjust metadata to proper format
    print("Found {} total samples in meta file {}".format(total_samples, path))
    age = [calc_age(photo_taken[i], dob[i]) for i in range(total_samples)]
    gender = [ "F" if x == 0 else "M" for x in gender ]

    data_out = []
    for i in range(total_samples):
        current_path = full_path[i][0]
        current_age = age[i]
        current_gender = gender[i]
        current_face_location = face_location[i]

        #if theres a second face detected, ignore
        if (~math.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue

        # If a face was not detected, ignore
        if (math.isinf(face_score[i]) is True):
            continue

        # If the confidence that a face was found is too low, ignore
        if face_score[i] < 1.0:
            continue

        if ~(0 <= current_age <= 100):
            continue

        if current_gender not in ["M", "F"]:
            continue

        full_image_path = os.path.join(path, current_path)
        if is_image_good(full_image_path) is False:
            continue


        data_out.append({
            "id": current_path,
            "gender":current_gender,
            "age":current_age
        })

    print("Kept total of {} good samples".format(len(data_out)))
    with open( os.path.join(path,"groundtruth.json"), 'w') as outfile:
        json.dump(data_out, outfile, indent=4, sort_keys=True)



if __name__ == "__main__":
    convert_mat_to_json("data/train/imdb_crop", 'imdb')