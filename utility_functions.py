
# PROGRAMMER: Abdullah Bahi
# DATE CREATED: 6/20/2020
# PURPOSE: Contains helper functions used train.py and predict.py scripts
#
# FUNCTIONS: 
#       - get_arguments(parser)
#       - get_prediction_arguments(parser)
#       - load_data(train_dir, valid_dir, T, train_batch_size, val_batch_size)
#       - process_image(image, T)
#       - predict(image_path, model, gpu=torch.cuda.is_available(), topk=5)
#       - print_predection(names_available, probs, classes, category_names_path)

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from torchvision import datasets, transforms
from PIL import Image


def get_arguments(parser):
    """
    adds the required CLI-arguments for train.py and returns args object.
    
    Parameters : 
            - parser : ArgumentParser object to which we will add and parse arguments
    
    Returns : 
            - in_args : ArgumentParser object with the parsed arguments
    """
    # get current working directory
    cwd = os.getcwd()
    
    # Add Positional arguments
    parser.add_argument('data_dir', type=str, default = 'flowers', help = 'directory of training and validation data')

    # Add Optional arguments
    parser.add_argument('--save_dir', type=str, default = cwd, help = 'directory to save model checpoint.pth file')
    parser.add_argument('--arch', type=str, default = 'vgg19_bn', help = 'network architecture to be trained')
    parser.add_argument('-lr','--learning_rate', type=float, default = 0.001, help = 'learning rate used for training')
    parser.add_argument('-h_u','--hidden_units', type=int, default = 1024, help = 'number of hidden units in classifier hidden layer')
    parser.add_argument('--epochs', type=int, default = 10, help = 'number of training epochs')
    parser.add_argument('--gpu', action="store_true", default = False, help = 'use GPU while trainig ')
    
    # Parse arguments
    in_args = parser.parse_args()
    
    return in_args

def get_prediction_arguments(parser):
    """
    adds the required CLI-arguments for predict.py and returns args object.
    
    Parameters : 
            - parser : ArgumentParser object to which we will add and parse arguments
    
    Returns : 
            - in_args : ArgumentParser object with the parsed arguments
    """
    # get current working directory
    cwd = os.getcwd()
    
    # Add Positional arguments
    parser.add_argument('img_path', type=str, help = 'path to the input image')
    parser.add_argument('chechpoint', type=str, help = 'path to the checkpoint.pth file')

    # Add Optional arguments
    parser.add_argument('--top_k', type=int, default = 1, help = 'number of top predicted classes to be returned')
    parser.add_argument('--category_names', type=str, default = "", help = 'path to json file used for mapping of categories                                                                                             to real names')
    parser.add_argument('--gpu', action="store_true", default = False, help = 'use GPU while predicting ')
    
    # Parse arguments
    in_args = parser.parse_args()
    
    return in_args


def load_data(train_dir, valid_dir, T, train_batch_size=64, val_batch_size=32):
    
    """
    loads training and validation data into a list of dataloaders.
    
    Parameters:
        - train_dir : directory of training data
        - valid_dir : directory of Validation data
        - train_batch_size : batch size of training set dataloader, default = 64
        - val_batch_size : batch size of validation set dataloader, default = 32
        - T : Dictionary of desired transforms with transform names as keys
    
    Returns:
        - dataloaders : a list of dataloaders with
            - training set dataloader (dataloaders[0])
            - validation set dataloader (dataloaders[1])
        - class_to_idx : mapping from class categories to indeces
    """                
        
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(T['RandomRotation']),
                                          transforms.RandomResizedCrop(T['RandomResizedCrop']),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(T['mean'], 
                                                                   T['std'])
                                         ])
    
    val_transforms = transforms.Compose([transforms.Resize(T['Resize']),
                                          transforms.CenterCrop(T['CenterCrop']),
                                          transforms.ToTensor(),
                                          transforms.Normalize(T['mean'], 
                                                                   T['std'])
                                         ])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_data = datasets.ImageFolder(valid_dir, transform=val_transforms)
    
    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = [torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True),
                   torch.utils.data.DataLoader(val_data, batch_size=val_batch_size)]
                      
    return dataloaders, train_data.class_to_idx


def process_image(image, T):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    imge = Image.open(image)
    imge = imge.resize((T['Resize'],T['Resize']))
    value = 0.5 * (T['Resize'] - T['CenterCrop'])
    imge = imge.crop((value,value,T['Resize']-value,T['Resize']-value))
    imge = np.array(imge) / (T['Resize'] - 1)

    mean = np.array(T['mean'])
    std = np.array(T['std'])
    imge = (imge - mean) / std

    return imge.transpose(2,0,1)


def predict(image_path, model, T, class_to_idx, gpu=torch.cuda.is_available(), topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # move the model to GPU memory if available
    device = torch.device("cuda" if gpu else "cpu")
    model.to(device)
    model.eval()
    
    image = process_image(image_path, T)
    image = torch.from_numpy(np.array([image])).float()
    image = image.to(device)
    
    output = model.forward(image)
    output = torch.exp(output)
    
    probs, indeces = output.topk(topk)
    probs = probs.tolist()[0]
    indeces = indeces.tolist()[0]
        
    classes = []
            
    for index in indeces:
        for element in list(class_to_idx.items()):
            if index == element[1]:
                classes.append(element[0])
        
    return probs, classes

def print_predection(names_available, probs, classes, category_names_path):
    
    if names_available:
        # Label mapping
        with open(category_names_path, 'r') as f:
            cat_to_name = json.load(f)
        
        # Get class names 
        class_names = []
        for class_ in classes:
            for key in cat_to_name:
                if key == class_:
                    class_names.append(cat_to_name[key])
        print("     Predicted Classes                   Class Scores")
        
        for index in range(len(class_names)):
            print("    ", class_names[index], "                          ", "{:.2f}%".format(probs[index]*100))
    else:
        # print predicted classes as numbers
        print("     Predicted Classes                Class Scores")
        
        for index in range(len(classes)):
            print("        ", classes[index], "                            ", "{:.2f}%".format(probs[index]*100))
        
                              
    