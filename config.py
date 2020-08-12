
# PROGRAMMER: Abdullah Bahi
# DATE CREATED: 6/20/2020
# PURPOSE: Contains configurations for loading data and training the model
#
# VARIABLES:
#       - transforms_dict       : dictionary of desired transformations on training and validation sets
#       - train_batch_size      : training dataset batch size 
#       - val_batch_size        : validation dataset batch size 
#       - num_classes           : number of classes (classifier output)
#       - calssifier_input_size : number of inputs to the classifier(output of feature detector),
#                                 this is found in the architecture documentation
#       - dropout               : drop out probability, set to to (1) to turn off Dropout
#       - train_with_validation : if true, use validation while training, otherwise, training is done without validation
#       - print_every           : print training loss and training accuracy every (print_every) batches
#       - criterion             : loss function
#       -

from torch import nn

class config():
    def __init__(self):
        """
        
        """
         ## Data specs
        # Define desired transformations on training and validation sets
        self.transforms_dict = {'RandomRotation':30,             # for training set only
                                'RandomResizedCrop':224,         # for training set only
                                'mean':[0.485, 0.456, 0.406],    # for both
                                'std':[0.229, 0.224, 0.225],     # for both
                                'Resize':256,                    # for validation set only
                                'CenterCrop':224                 # for validation set only
                                }
        
        # Define batch sizes
        self.train_batch_size = 64
        self.val_batch_size = 32
        
         ## Network specs
        # Define number of output classes
        self.num_classes = 102
        # Define number of inputs to the classifier(output of feature detector),
        # this is found in the architecture documentation
        self.calssifier_input_size = 25088
        # Define drop out probability, set to to (1) to turn off Dropout
        self.dropout = 0.5
                
         ## Training specs
        # if true, use validation while training
        self.train_with_validation = True
        # print training loss and training accuracy every (print_every) batches 
        self.print_every = 20
        # define loss function
        self.criterion = nn.NLLLoss()
        
        
        
        
        
        