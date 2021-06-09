## Instructions for how to create a text annotated image dataset and run AttnGAN on it
- Make descriptions for each image with manual_annotation.py
- Create the text files for each image with text_files_from_annotation.py
- Make the dataset pickle files
- Make the yml files for pretraining DAMSM and for training the GAN
- Pretrain the DAMSM with python pretrain_DAMSM.py. Model is stored in output
- Train the AttnGAN with python main.py. 
Move text and image encoders from output/^date^/Model to DAMSMencoders/^dataset^/
mv image_encoder200.pth ../../../DAMSMencoders/dilbert-3/

- Generate descriptions
Move attn model to /models
Add example captions to example_captions.txt
Find generated captions in models/

cp netG_epoch_150.pth ../../../models/dilbert-3_attn.pth

Download models
gcloud compute scp --project gan-g-shit --zone us-west1-b --recurse deeplearning-2-vm:~/new-gan-stuff/AttnGAN/models/dilbert-2_attn.pth
gcloud compute scp --project gan-g-shit --zone us-west1-b --recurse deeplearning-2-vm:~/new-gan-stuff/AttnGAN/DAMSMencoders/dilbert-2/text_encoder150.pth attnGAN_gen

upload exampletxt 

gcloud compute scp --project gan-g-shit --zone us-west1-b --recurse example_captions.txt deeplearning-2-vm:~/new-gan-stuff/AttnGAN/data/dilbert-3


#copy results
gcloud compute scp --project gan-g-shit --zone us-west1-b --recurse deeplearning-2-vm:~/new-gan-stuff/AttnGAN/models/dilbert-3 attnGAN_gen2.1

gcloud compute scp --project gan-g-shit --zone us-west1-b --recurse deeplearning-2-vm:~/new-gan-stuff/AttnGAN/models/dilbert-3 attnGAN_gen2.1/