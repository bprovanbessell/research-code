import json
import random
import glob
import os
import re
from shutil import copyfile, move

def make_test_captions(annotations_json, txt_folder):

    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)

    test_annotes = []

    left_over = []

    with open(annotations_json) as annotations_file:
        annotations_dict = json.load(annotations_file)
        for image_name in annotations_dict.keys():
            annotations = annotations_dict[image_name]

            annotes = random.choices(annotations, k=3)

            test_annotes.append(annotes[0])
            test_annotes.append(annotes[1])

            left_over.append(annotes[2])

    if len(test_annotes) < 2048:
        print(len(test_annotes))
        to_add = 2048 - len(test_annotes)
        test_annotes.extend(left_over[0:to_add])

        print(to_add)

    batch_size = 8

    file_names_object = open(txt_folder + "example_filenames.txt", "w+")

    for b in range(0, int(len(test_annotes)/batch_size)):
        file_name = txt_folder + "test_caption_{}.txt".format(b)

        file_names_object.write("test_captions/test_caption_{}".format(b))
        file_names_object.write("\n")

        file_object = open(file_name, "w+")

        i_1 = b * batch_size
        i_2 = (b+1) * batch_size

        annote_split = test_annotes[i_1: i_2]
        for annotation in annote_split:
            file_object.write(annotation)
            file_object.write("\n")
        file_object.close()


def test_photo_extractor(generated_folder, out_folder):
    all_images = glob.glob(generated_folder + "test_caption*/*_g2.png")

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for i, path in enumerate(all_images):

        dest = path.replace(generated_folder, out_folder)
        dest = re.sub(r'(?is)/test_caption(.*?)/', '/', dest)
        dest = dest.replace(".png", "_" + str(i) + ".png")
        move(path, dest)

def make_until_2048(im_folder):
    all_images = glob.glob(im_folder + "*.png")

    l = len(all_images)

    ll = 0

    for i, path in enumerate(all_images):
        if (ll > 2048):
            break
        else:
            ll = l + i
            dest = path.replace(".png", str(ll) + ".png")
            copyfile(path, dest)

if __name__ == "__main__":
    annotations_js = "data/dilbert/dilbert_annotations_3.json"
    txt_path = "data/dilbert/dil3_gen/test_captions/"

    gen_path = "data/dilbert/dil3_gen/fidtest/gen3_200/"
    out_gen_path = "data/dilbert/dil3_gen/fidtest/test_gen/final_gen3_200/"

    images_path = "data/dilbert/dil3_gen/fidtest/images/001.dilbert_3/"

    # make_test_captions(annotations_js, txt_path)

    test_photo_extractor(gen_path, out_gen_path)

    # make_until_2048(images_path)
