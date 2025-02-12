# PokedexML
This is an image classifier that was made to classify the original 151 Pokemon. It includes a CNN that I built to train the data and a testing file to test the data as well.

## How To Use
To use this program, you put the path of the folder where the data is located in the "data_path" variable in training.py and train the model. Then, using testing.py you can either test the data in batches or take paths of images, put them in the image_path array and uncomment the code where it says "For testing single images". The clean_files.py file is used to remove any corrupted images in the data folder.

## Tech Used
Python and PyTorch were used in this project. I researched on what to use for a Convolutional Neural Network for an image classifier and created my own CNN and was able to train and test it as well.

## Lessons Learned
I learned about writing CNN's for image classifiers and how to test single images for those classifiers.

## References
I used this Kaggle dataset to train and test on: https://www.kaggle.com/datasets/echometerhhwl/pokemon-gen-1-38914
