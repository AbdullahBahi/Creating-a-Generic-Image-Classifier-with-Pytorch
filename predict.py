
# PROGRAMMER: Abdullah Bahi
# DATE CREATED: 6/21/2020
# PURPOSE: Uses a trained network to predict the class for a given input image.
#
# USAGE: predict.py img_path chechpoint [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES] [--gpu]
#       
# Example call:
#    python predict.py input checkpoint --top_k 5 --category_names cat_to_name.json 

import argparse
import time

# import helper files
import utility_functions
import model_utilities 
import config

parser = argparse.ArgumentParser()

# Gets parser arguments
in_args = utility_functions.get_prediction_arguments(parser = parser)

print("System message: Initializing environment ...\n")

# Extract arguments
img_path = in_args.img_path
chechpoint_path = in_args.chechpoint
top_k = in_args.top_k
category_names_path = in_args.category_names
gpu = in_args.gpu

# Create configurations object to extract prediction settings
configs = config.config()

# load the model for predection (extract only required information)
_, _, _, _, _, _, class_to_idx, _, model = model_utilities.load_checkpoint(chechpoint_path)

print("System message: Environment Initialized Successfully.\n")

print("System message: Predicting the top {} classes for the given image ...\n".format(top_k))

# Predict the top (k) classes
probs, classes = utility_functions.predict(image_path = img_path,
                                           model = model,
                                           T = configs.transforms_dict,
                                           class_to_idx = class_to_idx,
                                           gpu = gpu,
                                           topk = top_k
                                          )
print("System message: The top {} classes for the given image are predictid.\n".format(top_k))

print("System message: Printing Output ...\n")

time.sleep(2)

# Check if the argument 'category_names' is passed
names_available = False
if category_names_path != parser.get_default('category_names'):
            names_available = True

utility_functions.print_predection(names_available = names_available,
                                   probs = probs,
                                   classes = classes,
                                   category_names_path = category_names_path,
                                  )




