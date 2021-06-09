import pickle


if __name__ == "__main__":

    classinfo = pickle.load(open("../../AttnGAN/data/birds/test/class_info.pickle", "rb"))
    filenames = pickle.load(open("../../AttnGAN/data/birds/test/filenames.pickle", "rb"))

    captions = pickle.load(open("../../AttnGAN/data/birds/captions.pickle", "rb"))

    filenames2 = pickle.load(open("../../pyth3/AttnGAN/data/dilbert-attn/test/filenames.pickle", "rb"))


    print(classinfo)
    print("--------")
    print(filenames)
    print(filenames2)

    print(type(captions))

    # print(captions[0:10])

    trainclassinfo_dil = [1]*60
    trainfilenameinfo_dil = []