import json
import random


def convert_to_multilabels(annotations_json, out_json):
    # print("Available characters:\n"
    #       "Dilbert: 1\n"
    #       "Dogbert: 2\n"
    #       "Boss: 3\n"
    #       "CEO: 4\n"
    #       "Wolly: 5\n"
    #       "Alice: 6\n"
    #       "Catbert: 7\n"
    #       "Asok: 8\n"
    #       "Secretary: 10\n"
    #       "Office guy: 11\n"
    #       "Big hair secretary: 12\n"
    #       "Non-recurring/main Character: 9\n"
    #       "Building: 0")
    #
    # # building as character
    # print("Availble background colours:\n"
    #       "yellow: y\n"
    #       "green: g:\n"
    #       "purple: p\n"
    #       "blue: b\n"
    #       "pink: pi\n"
    #       "white:")

    # map character label to vector index

    annote_to_label = {"0": 0,
                       "1": 1,
                       "2": 2,
                       "3": 3,
                       "5": 4,
                       "6": 5,
                       "7": 6,
                       "8": 7,
                       # "10": 8,
                       # "11": 9,
                       # "12": 10,
                       #
                       # "g": 11,
                       # "b": 12,
                       # "p": 13,
                       # "pi": 14,
                       # "y": 15,
                       # "w": 16
                       "g": 8,
                       "b": 9,
                       "p": 10,
                       "pi": 11,
                       "y": 12,
                       # "w": 13
                       }

    res = {}

    with open(annotations_json) as jsfile:
        annotations_dict = json.load(jsfile)

        for fn in annotations_dict.keys():
            annotes = annotations_dict[fn]

            chars = annotes["Characters"]
            colour = annotes["Colour"].replace(" ", "")

            # label_vector = [0]*17
            label_vector = [0] * 13

            for char in chars:
                label_vector[annote_to_label[char]] = 1

            label_vector[annote_to_label[colour]] = 1

            res[fn] = label_vector

    with open(out_json, "w+") as out_file:
        json.dump(res, out_file)


def convert_to_colour_labels(annotations_json, out_json):
    annote_to_label = {"0": 0,
                       "1": 1,
                       "2": 2,
                       "3": 3,
                       "5": 4,
                       "6": 5,
                       "7": 6,
                       "8": 7,
                       "10": 8,
                       "11": 9,
                       "12": 10,

                       "g": 11,
                       "b": 12,
                       "p": 13,
                       "pi": 14,
                       "y": 15,
                       "w": 16
                       }

    res = {}

    with open(annotations_json) as jsfile:
        annotations_dict = json.load(jsfile)

        for fn in annotations_dict.keys():
            annotes = annotations_dict[fn]

            chars = annotes["Characters"]
            colour = annotes["Colour"].replace(" ", "")

            label_vector = [0] * 6

            label_vector[(annote_to_label[colour] - 11)] = 1

            res[fn] = label_vector

    with open(out_json, "w+") as out_file:
        json.dump(res, out_file)


def convert_to_char_multilabels(annotations_json, out_json):
    annote_to_label = {"0": 0,
                       "1": 1,
                       "2": 2,
                       "3": 3,
                       "5": 4,
                       "6": 5,
                       "7": 6,
                       "8": 7,
                       "10": 8,
                       "11": 9,
                       "12": 10,

                       "g": 11,
                       "b": 12,
                       "p": 13,
                       "pi": 14,
                       "y": 15,
                       "w": 16
                       }

    res = {}

    with open(annotations_json) as jsfile:
        annotations_dict = json.load(jsfile)

        for fn in random.shuffle(list(annotations_dict.keys())):
            annotes = annotations_dict[fn]

            chars = annotes["Characters"]
            colour = annotes["Colour"].replace(" ", "")

            # dont take the less seen characters.
            conditions = conditions = [
                "10" in chars, "11" in chars, "12" in chars
            ]

            if not (any(conditions)):
                label_vector = [0] * 8

                for char in chars:
                    label_vector[annote_to_label[char]] = 1

                res[fn] = label_vector

    with open(out_json, "w+") as out_file:
        json.dump(res, out_file)


def split_data(labels_json, out_folder):
    train_dict = {}
    test_dict = {}

    split_ratio = 0.8

    with open(labels_json) as jsfile:
        labels_dict = json.load(jsfile)

        l = len(labels_dict)

        train_r = int(split_ratio * l)

        keys = list(labels_dict.keys())

        train_keys = keys[0:train_r]
        test_keys = keys[train_r:]

        train_dict = {key: labels_dict[key] for key in train_keys}
        test_dict = {key: labels_dict[key] for key in test_keys}

        train_file = out_folder + "train_equal_labels.json"
        test_file = out_folder + "test_equal_labels.json"

        with open(train_file, "w+") as tr_file:
            json.dump(train_dict, tr_file)

        with open(test_file, "w+") as test_file:
            json.dump(test_dict, test_file)


if __name__ == "__main__":
    annotations_json = "../data/dilbert/annotated-jsons/resized_char_and_colour_0:3000_equal.json"

    out_json = "../data/dilbert/annotated-jsons/char_colour_0:3000_equals_labels.json"

    convert_to_multilabels(annotations_json, out_json)

    # convert_to_singlelabels(annotations_json, out_json)

    split_data(out_json, "../data/dilbert/annotated-jsons/")
