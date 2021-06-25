import json
import glob
from shutil import move, copy
import pickle
import random

# get the right files
# parse them into json with same filenames
import os


def parse_to_json(txt_path, out_json):
    with open(txt_path, encoding="utf-8", errors='ignore') as f:
        lines = f.readlines()

        # get transcriptions from 2007 06 11 onwards
        lines = lines[6455:]

        result = {}

        l = 0

        while (l < len(lines) -2):
            print(l)

            line = lines[l]

            date = line[0:6]

            i = 1

            next_lines = []
            print(date)

            while lines[l + i][0:6] == date:
                next_lines.append(lines[l + i])
                i = i + 1

            year = line[0:2]
            month = line[2:4]
            day = line[4:6]

            next_lines.insert(0, line)

            full_t = ""
            for n_line in next_lines:
                p_l = n_line[10:]
                p_l = p_l.strip('\n')

                full_t = full_t + p_l + " "

            # remove last space
            full_t = full_t[:-1]

            split_t = full_t.split(" - ")

            for j, t in enumerate(split_t):
                k = "20" + year + "-" + month + "-" + day + "_" + str(j)

                result[k] = t

            l = l + i

    with open(out_json, "w+") as out_file:
        json.dump(result, out_file)

def json_to_txt_from_images(transcriptions_json, image_paths, out_folder, images_folder):

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    with open(transcriptions_json) as transcriptions_file:
        t_dict = json.load(transcriptions_file)

    missing = []

    for impath in image_paths:

        key = impath.replace(".png", "")
        key = key.replace(images_folder, "")

        # print(key)

        txt_path = impath.replace(".png", ".txt")
        txt_path = txt_path.replace(images_folder, out_folder)

        if key in t_dict.keys():
            annotation = t_dict[key]

            file_object = open(txt_path, "w+")
            file_object.write(annotation)
            file_object.close()

        else:
            print(key)
            missing.append(key)

    res = {"missing": missing}
    print(res)

    with open("missing.json", "w+") as missing_file:
        json.dump(res, missing_file)

def fix_miss_annotated_images(missing_json, images_folder):

    with open(missing_json) as missing_file:
        missing_dict = json.load(missing_file)

        missing_list = missing_dict["missing"]

        for missing in missing_list:

            base_path = images_folder + missing[0:-1] + "*"

            images_list = sorted(glob.glob(base_path))

            for i, image_path in enumerate(images_list):
                # remove last digit

                # x.png
                new_image_path = image_path[0:-5] + str(i) + ".png"

                move(image_path, new_image_path)

def remove_missing(missing_json, images_folder):

    if not os.path.exists("data/dilbert/dilbert_transcribed/missing/"):
        os.makedirs("data/dilbert/dilbert_transcribed/missing/")

    with open(missing_json) as missing_file:
        missing_dict = json.load(missing_file)

        missing_list = missing_dict["missing"]

        for missing in missing_list:

            images_list = sorted(glob.glob(images_folder + missing + ".png"))

            for i, image_path in enumerate(images_list):
                # remove last digit
                new_image_path = image_path.replace(images_folder, "data/dilbert/dilbert_transcribed/missing/")
                # x.png
                move(image_path, new_image_path)


def make_train_test_pickles(images_list, image_folder, out_folder):

    length = len(images_list)
    tr_length = round(length * .75)

    file_paths = [fn.replace(image_folder, "001.dilbert_transcribed/").replace(".png", "") for fn in images_list]

    # ['001.Black_footed_Albatross/Black_Footed_Albatross_0046_18',
    print(file_paths[0])

    class_i = [1]*length

    file_paths_train = file_paths[0:tr_length]
    file_paths_test = file_paths[tr_length:]

    classi = [1] * length
    classi_train = classi[0:tr_length]
    classi_test = classi[tr_length:]

    print(length)

    if not os.path.exists(out_folder + "train/"):
        os.makedirs(out_folder + "train/")

    if not os.path.exists(out_folder + "test/"):
        os.makedirs(out_folder + "test/")

    pickle.dump(file_paths_train, open(out_folder + "train/filenames.pickle", "wb"))
    pickle.dump(file_paths_test, open(out_folder + "test/filenames.pickle", "wb"))

    pickle.dump(classi_train, open(out_folder + "train/class_info.pickle", "wb"))
    pickle.dump(classi_test, open(out_folder + "test/class_info.pickle", "wb"))


def make_examples(transcriptions_list, transcriptions_folder, out_folder, out_folder_old):

    # gen 2048
    fn_object = open(out_folder.replace("gen_captions/", "") + "example_filenames", "w+")
    file_names = transcriptions_list[0:2050]

    for fn in file_names:
        copy_to = fn.replace(transcriptions_folder, out_folder)
        copy(fn, copy_to)

        fn = fn.replace(out_folder_old, "gen_captions/").replace(".txt", "")

        fn_object.write(fn)
        fn_object.write("\n")
    fn_object.close()


def save_losses(g_losses, d_losses, epoch):
    import json

    with open("g_d_losses_{}.json".format(epoch), "w+") as js_file:
        res = {"g_losses": g_losses, "d_losses": d_losses}
        json.dump(res, js_file)


def copy_images_transcriptions(image_folder, image_paths, transcriptions_folder, transcriptions_paths, copy_to_im, copy_to_txt):

    new_im_path = image_paths[0].replace(image_folder, copy_to_im)
    new_tr_path = transcriptions_paths[0].replace(transcriptions_folder, copy_to_txt)

    if not os.path.exists(new_im_path):
        os.makedirs(new_im_path)
        os.makedirs(new_tr_path)

    for im_path in image_paths:
        new_im_path = im_path.replace(image_folder, copy_to_im)
        move(im_path, new_im_path)

    for tr_path in transcriptions_paths:
        new_tr_path = tr_path.replace(transcriptions_folder, copy_to_txt)
        move(tr_path, new_tr_path)


if __name__ == "__main__":
    transcriptions_path = "../data/dilbert/transcriptions.txt"
    out_json = "data/dilbert/transcriptions.json"

    # parse_to_json(transcriptions_path, out_json)
    image_folder = "data/dilbert/dilbert_transcribed_10k/images/001.dilbert_transcribed/"
    image_paths = sorted(glob.glob("data/dilbert/dilbert_transcribed_10k/images/001.dilbert_transcribed/*"))
    out_folder = "data/dilbert/dilbert_transcribed/"

    train_paths_im = image_paths[0:3000]

    test_paths = image_paths[3000:5200]

    # json_to_txt_from_images(out_json, image_paths, out_folder, image_folder)

    # fix_miss_annotated_images("missing.json", image_folder)

    # remove_missing("missing.json", image_folder)


    transcriptions_folder = "data/dilbert/dilbert_transcribed_10k/text/001.dilbert_transcribed/"
    transcriptions_list = sorted(glob.glob("data/dilbert/dilbert_transcribed_10k/text/001.dilbert_transcribed/*"))

    train_paths_txt = transcriptions_list[0:3000]

    gen_trans = transcriptions_list[-2100:]

    print(len(image_paths))

    out_folder_txt = out_folder + "text/001.dilbert_transcribed/"
    out_folder_im = out_folder + "images/001.dilbert_transcribed/"

    image_folder_3k = "data/dilbert/dilbert_transcribed/images/001.dilbert_transcribed/"
    image_paths_3k = sorted(glob.glob("data/dilbert/dilbert_transcribed/images/001.dilbert_transcribed/*"))

    # make_train_test_pickles(image_paths_3k, image_folder_3k, out_folder)

    out_folder_gen_captions = out_folder + "gen_captions/"
    out_folder_old = "data/dilbert/dilbert_transcribed_10k/text/001.dilbert_transcribed/"

    print(len(gen_trans))

    # copy_images_transcriptions(image_folder, train_paths_im, transcriptions_folder, train_paths_txt, out_folder_im, out_folder_txt)

    # transcriptions_folder = "data/dilbert/dilbert_transcribed/text/001.dilbert_transcribed/"
    # make_examples(gen_trans, transcriptions_folder, out_folder_gen_captions, out_folder_old)

