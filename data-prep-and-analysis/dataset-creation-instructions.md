## Instructions for how to create a text annotated image dataset and run AttnGAN on it
- Make descriptions for each image with manual_annotation.py
- Or, to create descriptions from labels, use annotation_to_description.py (you can first label images with manual_annotation.py)
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

e.g. - 
cp netG_epoch_150.pth ../../../models/dilbert-3_attn.pth

- Copy results from google cloud 

gcloud compute scp --project gan --zone us-west1-b --recurse deeplearning-2-vm:~/new-gan-stuff/AttnGAN/models/dilbert attnGAN

gcloud compute scp --project gan --zone us-west1-b --recurse deeplearning-2-vm:~/new-gan-stuff/AttnGAN/models/dilbert attnGAN/