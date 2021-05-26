import json
import cv2
import glob

def manually_annotate_character_colour(paths, out_file_path):
    result = {}

    for i, img_path in enumerate(paths):
        img = cv2.imread(img_path)
        cv2.imshow("image", img)
        cv2.waitKey(1)

        print("Available characters:\n"
              "1: Dilbert\n"
              "2: Dogbert\n"
              "3: Boss")

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

        # bboxes

        cc_list = char_and_colour.split(",")
        chars = cc_list[0].split(" ")
        colour = cc_list[1]

        result[img_path] = {"Characters": chars, "Colour": colour}
        cv2.destroyAllWindows()

    with open(out_file_path, "w+") as out_file:
        json.dump(result, out_file)


if __name__ == "__main__":
    imgs_path = "data/dilbert/resized/*"

    out_file = "data/dilbert/resized_char_and_colour.json"

    paths = glob.glob(imgs_path)

    paths = sorted(paths)

    selected = paths[0:10]

    manually_annotate_character_colour(selected, out_file)