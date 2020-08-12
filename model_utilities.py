
# PROGRAMMER: Abdullah Bahi
# DATE CREATED: 6/20/2020
# PURPOSE: Contains helper functions used train.py and predict.py scripts
#
# FUNCTIONS: 
#       - build_model(arch, input_size, hidden_units, num_classes, dropout)
#       - model_validate(model, dataloader, criterion, gpu)
#       - model_train(model, dataloader, criterion, optimizer, gpu, epochs, print_every)
#       - save_model(model, save_dir, arch, lr, train_batch_size, val_batch_size, epochs, optimizer, class_to_idx)
#       - load_checkpoint(filepath)
#   

import torch
import torchvision
from torch import nn
from collections import OrderedDict
from torchvision import models

def build_model(arch, input_size=25088, hidden_units=1024, num_classes=102, dropout=0.5):
    """
    Builds a model with the desierd architecture and with
    one-hidden layer classifier.
    
    Parameters :
            - arch : architecture of the pretrained model to be loaded
            - hidden_units : num of hidden units in the classifier's hidden layer
            - dropout : Drop out probability, set to 1 to cancel Dropout
            - num_classes : number of output classes
    Returns :
            - model : the modified pre-trained model
    """
    model = getattr(torchvision.models, arch)(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_units)),
                                            ('drop', nn.Dropout(p=dropout)),
                                            ('relu', nn.ReLU()),
                                            ('fc2', nn.Linear(hidden_units, num_classes)),
                                            ('output', nn.LogSoftmax(dim=1))
                                            ]))
 
    model.classifier = classifier
    
    return model


def model_validate(model, dataloader, criterion, gpu = torch.cuda.is_available()):
    
    accuracy = 0
    val_loss = 0
    
    # move the model to GPU memory if available
    device = torch.device("cuda" if gpu else "cpu")
    model.to(device)
    
    # Model in inference mode, dropout is off
    model.eval()
    
    for images, labels in dataloader:
        
        # move training data to GPU memory if available
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        
        val_loss += criterion(output, labels).item()

        ## Calculating the accuracy 
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == torch.exp(output).max(dim=1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()
        
    print(">>>> Validation Loss = {:.3f}     ".format(val_loss/len(dataloader)),
          "Validation accuracy = {:.2f}%".format(accuracy/len(dataloader)*100))
    print("\n")

    return val_loss/len(dataloader), accuracy/len(dataloader)


def model_train(model, dataloader, criterion, optimizer, gpu = torch.cuda.is_available(), epochs=10, print_every=20):
    
    steps = 0
    running_loss = 0
    
    # move the model to GPU memory if available
    device = torch.device("cuda" if gpu else "cpu")
    model.to(device)

    # Training loop
    for e in range(epochs):
    
        # Model in training mode, dropout is on
        model.train()
        
        print("\n processed batches: ",end="")
        
        for images, labels in dataloader:
        
            steps += 1
        
            # move training data to GPU memory if available
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()

            if steps % print_every == 0:
            
                 ## Calculating the accuracy 
                # Class with highest probability is our predicted class, compare with true label
                equality = (labels.data == torch.exp(output).max(dim=1)[1])
                # Accuracy is number of correct predictions divided by all predictions, just take the mean
                accuracy = equality.type_as(torch.FloatTensor()).mean()
            
                print(steps,"\n",
                      "Training Loss = {:.3f}     ".format(running_loss/print_every),
                      "Training accuracy = {:.2f}%".format(accuracy*100))
                print("\n")
                print("processed batches: ",end="")
            
                running_loss = 0
                
                
def save_model(model, save_dir, arch, lr, train_batch_size, val_batch_size, epochs, optimizer, class_to_idx):
    checkpoint = {'input_size': model.classifier.fc1.in_features,
                  'output_size': model.classifier.fc2.out_features,
                  'arch': arch,
                  'learning_rate': lr,
                  'train_batch_size': train_batch_size,
                  'val_batch_size': val_batch_size,
                  'classifier' : model.classifier,
                  'epochs': epochs,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': class_to_idx}

    torch.save(checkpoint, save_dir+'/checkpoint.pth')
    
    
def load_checkpoint(filepath):
    
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    checkpoint = torch.load(filepath, map_location=map_location)

    input_size = checkpoint["input_size"]
    output_size = checkpoint["output_size"]
    learning_rate = checkpoint["learning_rate"]
    train_batch_size = checkpoint["train_batch_size"]
    val_batch_size = checkpoint["val_batch_size"]
    epochs = checkpoint['epochs']
    class_to_idx = checkpoint['class_to_idx']
    optimizer_state = checkpoint['optimizer']
    
    # Build the model
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    
    return input_size, output_size, learning_rate, train_batch_size, val_batch_size, epochs, class_to_idx, optimizer_state, model

