import json
import cv2

def manually_extract_text(img_paths, out_file_path):
    """
    Annotate images by typing the description in the command line.
    :param img_paths: a list of paths to images for annotation.
    :param out_file_path: a path to .json file that will be used to save the output.
    :return: None.
    """
    result = {}
    for i, img_path in enumerate(img_paths):
        # read and show the image
        img_path = "dcgan-dilbert/dilbert-data/dilbert/" + img_path

        img = cv2.imread(img_path)
        cv2.imshow("image", img)
        cv2.waitKey(1)

        # get text from console
        result[img_path] = input(f"{i + 1}/{len(img_paths)}:")
        cv2.destroyAllWindows()

    # dump results to json file
    with open(out_file_path, "w+") as out_file:
        json.dump(result, out_file)

if __name__ == '__main__':

    with open("../data/dilbert/splitted_paths.json") as paths_file:
        paths_list = json.load(paths_file)

        paths_list = sorted(paths_list)

        # print(paths_list[0:100])

        out_file = "../dilbert_annotations_2.json"

        manually_extract_text(paths_list[0:300], out_file)
    print("input paths and stuff")