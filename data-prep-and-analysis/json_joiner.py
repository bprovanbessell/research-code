import json

if __name__ == '__main__':

    d1, d2, d3, d4, d5, d6 = {} ,{}, {}, {}, {}, {}
    with open("data/dilbert/resized_char_and_colour_0:1500.json") as file1:
        d1 = json.load(file1)

    with open("data/dilbert/resized_char_and_colour_darwin.json") as file1:
        d2 = json.load(file1)

    # with open("data/dilbert/resized_char_and_colour_100:200.json") as file1:
    #     d3 = json.load(file1)
    #
    # with open("data/dilbert/resized_char_and_colour_300:412.json") as file1:
    #     d4 = json.load(file1)
    #
    # with open("data/dilbert/resized_char_and_colour_412:500.json") as file1:
    #     d5 = json.load(file1)
    #
    # with open("data/dilbert/resized_char_and_colour_500:750.json") as file1:
    #     d6 = json.load(file1)

    print(d1)
    d1.update(d2)
    # d1.update(d3)
    # d1.update(d4)
    # d1.update(d5)
    # d1.update(d6)

    with open("data/dilbert/resized_char_and_colour_0:2000.json", "w+") as out_file:
        json.dump(d1, out_file)

    print(len(d1.keys()))

