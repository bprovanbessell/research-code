# repreat examples of classes that are sparse, for both colour and character
# just the list of filenames

import json
from shutil import copy
import os

def duplicate_on_conditions(labels_json, out_json, image_folder, out_image_folder):

    if not os.path.exists(out_image_folder):
        os.makedirs(out_image_folder)

    with open(labels_json) as labels_json_file:
        labels_dict = json.load(labels_json_file)

        result = {}

        for fn in labels_dict:
            labels = labels_dict[fn]

            chars = labels["Characters"]
            colour = labels["Colour"]

            orig_image_file = os.path.join(image_folder, fn)

            # colours all even except for pink
            conditions = ["pi" == colour, "2" in chars, "5" in chars, "6" in chars, "7" in chars, "8" in chars, "0" in chars]
            # conditions = ["pi" == colour]

            if any(conditions):
                dup_name = fn.replace(".png", "_dup.png")

                result[dup_name] = labels

                out_image_file = os.path.join(out_image_folder, dup_name)

                copy(orig_image_file, out_image_file)

                if colour == "pi":
                    dup_name_2 = fn.replace(".png", "_dup1.png")
                    dup_name_3 = fn.replace(".png", "_dup2.png")
                    result[dup_name_2] = labels
                    result[dup_name_3] = labels

                    out_image_file = os.path.join(out_image_folder, dup_name_2)

                    copy(orig_image_file, out_image_file)

                    out_image_file = os.path.join(out_image_folder, dup_name_3)

                    copy(orig_image_file, out_image_file)

            result[fn] = labels
            out_image_file = os.path.join(out_image_folder, fn)

            copy(orig_image_file, out_image_file)

        with open(out_json, "w+") as out_file:
            json.dump(result, out_file)


if __name__ == "__main__":
    labels_filtered = "../data/dilbert/annotated-jsons/resized_char_and_colour_0:3000_filtered_basic_w.json"
    out_json = "../data/dilbert/annotated-jsons/resized_char_and_colour_0:3000_equal.json"
    image_folder = "../data/dilbert/resized_256_07-11"
    out_image_folder = "../data/dilbert/dilbert_equal/images/001.dilbert_equal"

    duplicate_on_conditions(labels_filtered, out_json, image_folder, out_image_folder)