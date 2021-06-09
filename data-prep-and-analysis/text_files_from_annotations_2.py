import json
from shutil import copyfile
import pickle

import cv2
import os

def mirror_images(out_folder, annotations_json, images_folder, mirror=False):

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    with open(annotations_json) as annotations_file:
        annotations_dict = json.load(annotations_file)
        for filename in annotations_dict.keys():
            annotation = annotations_dict[filename]

            file_path = images_folder + filename

            dst = out_folder + filename
                # we want to mirror these
            if mirror:
                originalImage = cv2.imread(file_path)
                mirroredimage = cv2.flip(originalImage, 1)

                cv2.imwrite(dst.replace(".png", "_mirror.png"), mirroredimage)
            copyfile(file_path, dst)


def make_file_for_annotations(out_folder, annotations_json, mirror=False):

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    with open(annotations_json) as annotations_file:
        annotations_dict = json.load(annotations_file)
        for image_name in annotations_dict.keys():
            annotations = annotations_dict[image_name]


            txt_name = image_name.replace(".png", ".txt")

            txt_path = out_folder + txt_name
            file_object = open(txt_path, "w+")

            for annotation in annotations:
                file_object.write(annotation)
                file_object.write("\n")
            file_object.close()

            if mirror:
                txt_path = txt_path.replace(".txt", "_mirror.txt")
                file_object = open(txt_path, "w+")
                for annotation in annotations:
                    file_object.write(annotation)
                    file_object.write("\n")
                file_object.close()

def make_pickles(out_folder, annotations_json, mirror=False):

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    file_paths = []

    with open(annotations_json) as annotations_file:
        annotations_dict = json.load(annotations_file)
        for image_name in annotations_dict.keys():
            annotation = annotations_dict[image_name]

            image_path = "001.dilbert_3/" + image_name
            dst = image_path.replace(".png", "")
            file_paths.append(dst)

            if mirror:
                dst = image_path.replace(".png", "_mirror")
                file_paths.append(dst)

    # file_paths_train = file_paths[0:60]
    print(file_paths[0])
    #
    # file_paths_test = file_paths[60:]
    #
    # classi = [1]*len(file_paths)
    # classi_train = classi[0:60]
    #
    # classi_test = classi[60:]

    length = len(file_paths)

    print(length)

    tr_length = round(length*.95)

    file_paths_train = file_paths[0:tr_length]
    file_paths_test = file_paths[tr_length:]

    classi = [1]*length
    classi_train = classi[0:tr_length]
    classi_test = classi[tr_length:]

    if not os.path.exists(out_folder+"train/"):
        os.makedirs(out_folder+"train/")

    if not os.path.exists(out_folder+"test/"):
        os.makedirs(out_folder+"test/")

    pickle.dump(file_paths_train, open(out_folder + "train/filenames.pickle", "wb"))
    pickle.dump(file_paths_test, open(out_folder+"test/filenames.pickle", "wb"))

    pickle.dump(classi_train, open(out_folder+"train/class_info.pickle", "wb"))
    pickle.dump(classi_test, open(out_folder+"test/class_info.pickle", "wb"))


if __name__ == "__main__":
    out_folder_t = "data/dilbert/dilbert-attn-3/text/001.dilbert_3/"
    annotations_js = "data/dilbert/dilbert_annotations_3.json"

    out_folder_i = "data/dilbert/dilbert-attn-3/images/001.dilbert_3/"

    out_folder_p = "data/dilbert/dilbert-attn-3/"

    images_folder = "data/dilbert/dilbert/cleared/"

    # mirror_images(out_folder_i, annotations_js, images_folder, True)
    # make_file_for_annotations(out_folder_t, annotations_js, True)
    # make_pickles(out_folder_p, annotations_js, True)
