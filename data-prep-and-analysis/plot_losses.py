import json
import matplotlib.pyplot as plt


def plot_gen_disc_from_file(json_file):

    with open(json_file) as json_file:
        d1 = json.load(json_file)
        g_losses = d1["g_losses"]
        d_losses = d1["d_losses"]

        plt.plot(g_losses)
        plt.plot(d_losses)

        plt.legend(["generator losses", "discriminator_losses"], loc='upper left')

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()


if __name__ == "__main__":
    fp = "data/attnGAN_gen/dilb_transcribed/json_losses_transcribed/g_d_losses_350.json"

    plot_gen_disc_from_file(fp)
