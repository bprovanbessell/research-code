from PIL import Image
import json
import os
import sys


# resize
# crop to centre

def resize(in_folder, file_list, size, out_folder):

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for file_path in file_list:

        file_path = file_path.replace("../../cleared-bubbles/dilbert/splitted", "cleared")

        path = in_folder + file_path

        if os.path.exists(path):

            image = Image.open(path)
            width, height = image.size

            if (width - height) < (0.1 * width):
                # image is square

                resized = image.resize((size, size))

                save_path = out_folder + file_path.replace("cleared", "")
                resized.save(save_path)


if __name__ == "__main__":
    with open("data/dilbert/dilbert/splitted_paths.json") as paths_file:
        paths_list = json.load(paths_file)

        paths_list = sorted(paths_list)

        in_folder = "data/dilbert/dilbert/"

        out_folder = "data/dilbert/resized/"

        to_resize = paths_list[0:5000]

        print("resizing")

        new_size = 256
        resize(in_folder, to_resize, new_size, out_folder)


