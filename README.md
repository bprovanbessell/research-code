### DescriptionGAN
Pytorch implementation for reproduction of results in paper Text-to-comic Generative Adversarial Network 
by Benjamin Provan-Bessell, under supervison of Lydia Chen and Zilong Zhao. See rp_text_to_comic_ben_provan-bessell.pdf 
for full details on motivation and results.

This repository contains 4 main parts:
* ComicGAN - A modified version of AttnGAN (https://github.com/taoxugit/AttnGAN) used to generate
comics from descriptions and dialogue. N.b. refer to https://github.com/bprovanbessell/AttnGAN for usage. The Comics CNN is also integrated as the image encoder, and is 
available as an option to use instead of Inception V3. See ReadMe in the according repository for use instructions.
* Data preparation scripts for formatting comic datasets to run with DescriptionGAN. 
See dataset-creation-instructions.md for further details.
* Python code used for multi-label classification of Dilbert comics. Please use the jupyter notebook for ease of use.
* Versions of DCGAN, WGAN GP, and StabilityGAN used to generate comics (see DCGAN models). Refer to https://github.com/LMescheder/GAN_stability for the StabilityGAN model.

Please note: Image datasets have been omitted according to terms and conditions regarding re-distribution
of Dilbert comcis (see https://dilbert.com/terms).
