## Instructions for how to create a text annotated image dataset and run AttnGAN on it
- Make descriptions for each image with manual_annotation.py
- Create the text files for each image with text_files_from_annotation.py
- Make the dataset pickle files
- Make the yml files for pretraining DAMSM and for training the GAN
- Pretrain the DAMSM with python pretrain_DAMSM.py. Model is stored in output
- Train the AttnGAN with python main.py. 
Move text and image encoders from output/^date^/Model to DAMSMencoders/^dataset^/
