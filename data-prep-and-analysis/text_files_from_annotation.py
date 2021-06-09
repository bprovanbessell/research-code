import json
from shutil import copyfile
import pickle

import cv2
import os


def make_file_for_annotations(out_folder, annotations_json, mirror=False):

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    with open(annotations_json) as annotations_file:
        annotations_dict = json.load(annotations_file)
        for path in annotations_dict.keys():
            annotation = annotations_dict[path]

            if annotation != "p" and "man" not in annotation:

                if "and" in annotation:
                    alist = annotation.split(" ")
                    a_2 = alist[2] + " " + alist[1] + " " + alist[0]

                else:
                    a_2 = annotation

                file_name = path.replace("dcgan-dilbert/dilbert-data/dilbert/splitted/", out_folder).replace(".png", ".txt")
                file_object = open(file_name, "w+")
                file_object.write(annotation)
                file_object.write("\n")
                file_object.write(a_2)
                file_object.close()

                if mirror:
                    file_name = path.replace("dcgan-dilbert/dilbert-data/dilbert/splitted/",
                                             out_folder).replace(".png", "_mirror.txt")
                    file_object = open(file_name, "w+")
                    file_object.write(annotation)
                    file_object.write("\n")
                    file_object.write(a_2)
                    file_object.close()


def get_annotated_images(out_folder, annotations_json, mirror=False):

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    with open(annotations_json) as annotations_file:
        annotations_dict = json.load(annotations_file)
        for path in annotations_dict.keys():
            annotation = annotations_dict[path]

            if annotation != "p" and "man" not in annotation:
                dst = path.replace("dcgan-dilbert/dilbert-data/dilbert/splitted/", out_folder)

                # we want to mirror these
                if mirror:
                    originalImage = cv2.imread(path)
                    mirroredimage = cv2.flip(originalImage, 1)

                    cv2.imwrite(dst.replace(".png", "_mirror.png"), mirroredimage)
                copyfile(path, dst)


def make_pickles(out_folder, annotations_json, mirror=False):

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    file_paths = []

    with open(annotations_json) as annotations_file:
        annotations_dict = json.load(annotations_file)
        for path in annotations_dict.keys():
            annotation = annotations_dict[path]

            if annotation != "p" and "man" not in annotation:
                dst = path.replace("dcgan-dilbert/dilbert-data/dilbert/splitted/", "001.dilbert_1/").replace(".png", "")
                file_paths.append(dst)

                if mirror:
                    dst = path.replace("dcgan-dilbert/dilbert-data/dilbert/splitted/", "001.dilbert_1/").replace(".png", "_mirror")
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

    tr_length = round(length*.8)

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
    out_folder_t = "dilbert-attn-2/text/001.dilbert_1/"
    annotations_js = "dilbert_annotations_2.json"

    out_folder_i = "dilbert-attn-2/images/001.dilbert_1/"

    out_folder_p = "dilbert-attn-2/"

    # make_file_for_annotations(out_folder_t, annotations_js, True)
    # get_annotated_images(out_folder_i, annotations_js, True)
    make_pickles(out_folder_p, annotations_js, True)
