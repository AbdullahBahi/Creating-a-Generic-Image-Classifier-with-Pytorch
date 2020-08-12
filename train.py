
# PROGRAMMER: Abdullah Bahi
# DATE CREATED: 6/20/2020
# PURPOSE: trains a classifier on an image dataset using a pre-trained network
# arcitecture and saves the model as a checkpoint.pth file.
#
# USAGE: train.py data_dir [-h] [--save_dir SAVE_DIR] [--arch ARCH] [-lr LEARNING_RATE]
#                [-h_u HIDDEN_UNITS] [--epochs EPOCHS] [--gpu]
#                
# EXAMPLE CALL:
#    python train.py /home/workspace/ImageClassifier/flowers --arch vgg19_bn -lr 0.001 -h_u 1024 --epochs 6 --gpu
#
# NOTE: you must edit the file named "config.py" first before using this file

import torch
import argparse
from torch import optim

# import helper files
import utility_functions
import model_utilities 
import config

parser = argparse.ArgumentParser()

# Gets parser arguments
in_args = utility_functions.get_arguments(parser = parser)

print("System message: Initializing environment ...\n")

# Extract arguments
data_dir = in_args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
save_dir = in_args.save_dir
arch = in_args.arch
learning_rate = in_args.learning_rate
hidden_units = in_args.hidden_units
epochs = in_args.epochs
gpu = in_args.gpu

# Create configurations object to extract training settings
configs = config.config()

# Loading training and validation datasets into dataloaders
dataloaders, class_to_idx = utility_functions.load_data(train_dir = train_dir,
                                                        valid_dir = valid_dir,
                                                        T = configs.transforms_dict,
                                                        train_batch_size = configs.train_batch_size,
                                                        val_batch_size = configs.val_batch_size)
trainloader = dataloaders[0]
validloader = dataloaders[1]

# Building the network
model = model_utilities.build_model(arch = arch,
                                    input_size = configs.calssifier_input_size,
                                    hidden_units = hidden_units,
                                    num_classes = configs.num_classes,
                                    dropout = configs.dropout)

# define optimizer
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

print("System message: Environment Initialized Successfully\n")

# Training the network 
if configs.train_with_validation :
    print("System message: Starting Training with validation and auto checkpoint-save every training epoch\n")    
    for e in range(epochs):
        ## train for one epoch
        model_utilities.model_train(model = model,
                                    dataloader = trainloader,
                                    criterion = configs.criterion,
                                    optimizer = optimizer,
                                    gpu = gpu,
                                    epochs = 1,
                                    print_every = configs.print_every)
    
        ## Do validation on the validation set
        print("\n\nProcess Message: {} training epochs out of {} finished, Validating the model ...".format(e+1, epochs))
        # Turn off gradients for validation, will speed up inference
        with torch.no_grad():
            val_loss, accuracy = model_utilities.model_validate(model = model,
                                                                dataloader = validloader,
                                                                criterion = configs.criterion)
        
        ## Save a checkpoint automaically every 1 epoch
        print("\n","Process Message: Validation done Successfully, Saving a Checkpoint ...")
        
        model_utilities.save_model(model = model,
                                   save_dir = save_dir,
                                   arch = arch,
                                   lr = learning_rate,
                                   train_batch_size = configs.train_batch_size,
                                   val_batch_size = configs.val_batch_size,
                                   epochs = e,
                                   optimizer = optimizer,
                                   class_to_idx = class_to_idx
                                  )
        print("Process Message: Check point saved Successfully in: ", save_dir+"/checkpoint.pth\n")
        
else:
    print("System message: Starting Training with auto checkpoint-save every training epoc, No Validation is done.h\n")  
    for e in range(epochs):
        ## train for one epoch
        model_utilities.model_train(model = model,
                                    dataloader = trainloader,
                                    criterion = configs.criterion,
                                    optimizer = configs.optimizer,
                                    gpu = gpu,
                                    epochs = 1,
                                    print_every = configs.print_every)
        
        ## Save a checkpoint automaically every 1 epoch
        print("\nProcess Message: Validation done Successfully, Saving a Checkpoint ...")
        
        model_utilities.save_model(model = model,
                                   save_dir = save_dir,
                                   arch = arch,
                                   lr = learning_rate,
                                   train_batch_size = configs.train_batch_size,
                                   val_batch_size = configs.val_batch_size,
                                   epochs = e,
                                   optimizer = configs.optimizer,
                                   class_to_idx = class_to_idx
                                  )
        print("Process Message: Check point saved Successfully in: ", save_dir+"/checkpoint.pth\n")
