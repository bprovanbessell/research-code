import json

from itertools import permutations, combinations
import random

def filter_on_conditions(labels_json, out_json):
    with open(labels_json) as labels_file:
        labels_dict = json.load(labels_file)

        print(len(labels_dict))

        result = {}

        for fn in labels_dict.keys():
            labels = labels_dict[fn]

            chars = labels["Characters"]
            labels["Colour"] = labels["Colour"].replace(" ", "")
            colour = labels["Colour"].replace(" ", "")

            if "" in chars:
                chars.remove("")

            # conditions
            conditions = ["9" in chars, "4" in chars, "blah" in chars, "-1" in chars, len(chars) > 3, "" in chars,
                          colour == "", len(chars) == 0, "b03" == colour,
                          "10" in chars, "11" in chars, "12" in chars,
                          colour == "w"
                          ]

            # conditions = ["9" not in chars, colour == ""]

            if not(any(conditions)):
                result[fn] = labels

        with open(out_json, "w+") as out_file:
            json.dump(result, out_file)


def convert_labels_to_annote(labels_json, out_json):
    with open(labels_json) as labels_file:
        labels_dict = json.load(labels_file)

        result = {}
        for fn in labels_dict.keys():
            labels = labels_dict[fn]

            chars = labels["Characters"]
            colour = labels["Colour"].replace(" ", "")

            chars_dict = {"1": "Dilbert",
                          "2": "Dog",
                          "3": "Boss",
                          "4": "CEO",
                          "5": "Wolly",
                          "6": "Alice",
                          "7": "Cat",
                          "8": "Asok",
                          "0": "Building"}

            colours_dict = {"b": "blue",
                            "g": "green",
                            "y": "yellow",
                            "p": "purple",
                            "pi": "pink",
                            "w": "white"}

            # remove non repeated and ceo
            characters = [chars_dict[c] for c in chars]
            colour = colours_dict[colour]

            result[fn] = shuffle_chars_and_colour(characters, colour)

        with open(out_json, "w+") as out_file:
            json.dump(result, out_file)

        print(len(result.keys()))

def shuffle_chars_and_colour(characters, colour):
    # shuffle ordering of characters
    # ordering of background

    # with max 3 characters, 6 permutations of character

    descriptions = []

    if len(characters) == 1:
        d1 = colour + " background with " + characters[0]
        d2 = characters[0] + " with " + colour + " background"

        descriptions.extend([d1, d2, d1, d2, d1, d2])

    elif len(characters) == 2:
        perm_chars = permutations(characters)

        for char_list in perm_chars:
            base_d = ""
            for character in char_list:
                base_d = base_d + character + " and "

            # remove last and
            base_d = base_d[0:-5]

            # add colour
            d1 = colour + " background with " + base_d
            d2 = base_d + " with " + colour + " background"

            descriptions.append(d1)
            descriptions.append(d2)
            descriptions.append(d1)
            descriptions.append(d2)

        descriptions = random.sample(descriptions, 6)

    else:
        perm_chars = permutations(characters)

        for char_list in perm_chars:
            base_d = ""
            for character in char_list:
                base_d = base_d + character + " and "

            # remove last and
            base_d = base_d[0:-5]

            # add colour
            d1 = colour + " background with " + base_d
            d2 = base_d + " with " + colour + " background"

            descriptions.append(d1)
            descriptions.append(d2)

        descriptions = random.sample(descriptions, 6)

    return descriptions

def count_colours(labels_json):
    with open(labels_json) as labels_file:
        labels_dict = json.load(labels_file)

        length = len(labels_dict)
        print(length)

        colours_dict = {"b": 0,
                        "g": 0,
                        "y": 0,
                        "p": 0,
                        "pi": 0,
                        "w": 0}

        chars_dict = {"1": 0,
                      "2": 0,
                      "3": 0,
                      "4": 0,
                      "5": 0,
                      "6": 0,
                      "7": 0,
                      "8": 0,
                      "9": 0,
                      "10": 0,
                      "11": 0,
                      "12": 0,
                      "0": 0}

        for fn in labels_dict.keys():
            labels = labels_dict[fn]

            chars = labels["Characters"]
            colour = labels["Colour"].replace(" ", "")

            colours_dict[colour] = colours_dict[colour] + 1

            for c in chars:
                chars_dict[c] = chars_dict[c] + 1



        p_dict = {k: v/length for k,v in colours_dict.items()}
        p_dict_c = {k: v/length for k,v in chars_dict.items()}

        print(colours_dict)

        print(chars_dict)

        print(p_dict)
        print(p_dict_c)


if __name__ == "__main__":
    labels_json = "data/dilbert/annotated-jsons/resized_char_and_colour_0:3000_filtered.json"

    labels_filtered = "data/dilbert/annotated-jsons/resized_char_and_colour_0:3000_equal.json"

    out_json = "data/dilbert/annotated-jsons/dilbert_annotations_equal.json"

    # filter_on_conditions(labels_json, labels_filtered)
    convert_labels_to_annote(labels_filtered, out_json)

    count_colours(labels_filtered)
    # for c in permutations(l):
    #     for ch in c:
    #         print(ch)

    # print(random.sample(l, 6))