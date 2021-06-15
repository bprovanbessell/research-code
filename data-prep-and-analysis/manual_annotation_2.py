import json
import cv2
import glob
import os

def manually_annotate_character_colour(paths, out_file_path):
    result = {}

    print(len(paths))

    for i, img_path in enumerate(paths):
        img = cv2.imread(img_path)
        cv2.imshow("image", img)
        cv2.waitKey(1)

        print("Available characters:\n"
              "Dilbert: 1\n"
              "Dogbert: 2\n"
              "Boss: 3\n"
              "CEO: 4\n"
              "Wolly: 5\n"
              "Alice: 6\n"
              "Catbert: 7\n"
              "Asok: 8\n"
              "Secretary: 10\n"
              "Office guy: 11\n"
              "Big hair secretary: 12\n"
              "Non-recurring/main Character: 9\n"
              "Building: 0")

        #         secretary
        # boss's wife
        # ratbert
        # office worker
        # elbonian

        # what to do about non-recurring characters?
        # building as character
        print("Availble background colours:\n"
              "yellow: y\n"
              "green: g:\n"
              "purple: p\n"
              "blue: b\n"
              "pink: pi\n"
              "white:")
        # brown??
        # split colours

        char_and_colour = input(f"{i + 1}/{len(paths)}: character character etc., background_colour: ")

        try:
            cc_list = char_and_colour.split(",")
            chars = cc_list[0].split(" ")
            colour = cc_list[1]

            filename = os.path.basename(img_path)

            # json will have only filename as key instead of filepath as Ben had previously
            result[filename] = {"Characters": chars, "Colour": colour}

        except:
            print("you messed up, saving results til now")
            with open(out_file_path, "w+") as out_file:
                json.dump(result, out_file)

            raise Exception("you messed up, saving results til now")

        # bboxes

        # cc_list = char_and_colour.split(",")
        # chars = cc_list[0].split(" ")
        # colour = cc_list[1]

        # result[img_path] = {"Characters": chars, "Colour": colour}
        cv2.destroyAllWindows()

    with open(out_file_path, "w+") as out_file:
        json.dump(result, out_file)


if __name__ == "__main__":
    imgs_path = "data/dilbert/resized_256_07-11/*"

    out_file = "data/dilbert/resized_char_and_colour_2500_92s_res.json"

    paths = glob.glob(imgs_path)

    paths = sorted(paths)

    selected = paths[2750:3000]

    print(selected)

    # manually_annotate_character_colour(selected, out_file)

    with open("data/dilbert/annotated-jsons/resized_char_and_colour_2500_9s.json") as jsfile:
        d = json.load(jsfile)

        file_paths = ["data/dilbert/resized_256_07-11/" + k for k in d.keys()]

        manually_annotate_character_colour(file_paths[76:], out_file)