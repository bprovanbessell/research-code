from PIL import Image
import json
import os
import sys
import glob

from shutil import move


# resize
# crop to centre

def resize(in_folder, file_list, size, out_folder):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for path in file_list:

        # file_path = file_path.replace("../../cleared-bubbles/dilbert/splitted", "cleared")

        # path = in_folder + file_path

        if os.path.exists(path):

            image = Image.open(path)
            width, height = image.size

            if (width - height) < (0.1 * width):
                # image is square

                resized = image.resize((size, size))

                save_path = path.replace(in_folder, out_folder)
                resized.save(save_path)


def move_to(fns, inf, outf):
    if not os.path.exists(outf):
        os.makedirs(outf)

    for path in fns:
        newpath = path.replace(inf, outf)

        move(path, newpath)


if __name__ == "__main__":
    with open("data/dilbert/dilbert/splitted_paths.json") as paths_file:
        paths_list = json.load(paths_file)

        paths_list = sorted(paths_list)

        in_folder = "data/dilbert/dilbert/cleared/"
        # in_folder = "data/dilbert/dil3_gen/fidtest/images/001.dilbert_3/"

        out_folder = "data/dilbert/resized_256_all/"
        # out_folder = "data/dilbert/dil3_gen/fidtest/images/dil3train_resized/"

        # to_resize = paths_list[0:5000]

        # to_resize = glob.glob("data/dilbert/dil3_gen/fidtest/images/001.dilbert_3/*.png")
        to_resize = glob.glob(("data/dilbert/dilbert/cleared/*.png"))

        to_move = glob.glob("data/dilbert/resized_256_all/*.png")

        to_move = sorted(to_move)
        to_move = to_move[0:10000]
        print("resizing")

        new_out = "data/dilbert/resized_256_10k/"

        new_size = 256
        # new_size = 64
        # resize(in_folder, to_resize, new_size, out_folder)

        print(to_move[0:10])
        move_to(to_move, out_folder, new_out)
