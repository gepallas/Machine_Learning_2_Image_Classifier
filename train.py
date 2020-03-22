################################################################
# Import Libraries
################################################################
import torch
import argparse
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
import pandas as pd
import numpy as np
import seaborn as sns
import json


################################################################
# Argument Parser
################################################################
def argument_parser():
    
    parser = argparse.ArgumentParser(description='Pass parameters to train a NN model on an image dataset')
    
    parser.add_argument('--architecture', type=str, default='vgg16',
                        help='Choose architecture as str between vgg and densenet architectures - default: "vgg16"')
    
    parser.add_argument('--hidden_layer', type=int, default=[1024, 512] , nargs='+',
                        help='Pass 2 int numbers for the hidden layer - default: 1024 512')
    
    parser.add_argument('--output_layer', type=int, default=102,
                        help='Pass output layer as int - default: 102')
    
    parser.add_argument('--learning_rate', type=float, default=0.0015,
                        help='Pass learning rate as float - default: 0.0015')
    
    parser.add_argument('--epochs', type=int, default=4,
                        help='Pass number of epochs as int - default: 4')
    
    parser.add_argument('--gpu', type=bool, default=True,
                        help='GPU is selected - default: True. Turn off GPU to switch to CPU by typing: False.')
    
    parser.add_argument('data_directory', type=str,
                        help='Pass your data directory as str')
    
    args = parser.parse_args()
    return args


################################################################
# Data Transformation and Loading
################################################################
def data_transform_load(train_dir, test_dir, valid_dir):
    
    # Transforms for train, test/valid datasets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_valid_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, test_valid_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, test_valid_transforms)

    # Define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64)
    
    return trainloader, testloader, validloader, train_datasets


################################################################
# Load Pre-trained Model and Build Classifier
################################################################

def gpu_selection(gpu):
    
    # Select between CPU and GPU availability
    if gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print('Running on GPU.')
        else:
            device = torch.device("cpu")
            print('GPU is not available. Running on CPU.')
    else:
        device = torch.device("cpu")
        print('Running on CPU')
    
    return device


################################################################
# Load Pre-trained Model and Build Classifier
################################################################
def build_model(architecture, learning_rate, hidden_layer, device, output_layer):
    
    # Load model
    method_to_call = getattr(models, architecture)
    model = method_to_call(pretrained=True)

    # Freeze parameters of pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Set input size for different model architectures. Output size is pre-defined from user.
    output_size = output_layer
    if architecture[:-2] == 'vgg':
        input_size = 25088        
    elif architecture[:-3] == 'densenet':
        input_size = 1024        
        
    # Build classifier
    model.classifier = nn.Sequential(nn.Linear(input_size, hidden_layer[0]),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_layer[0], hidden_layer[1]),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_layer[1], output_size),
                                     nn.LogSoftmax(dim=1))

    # Define loss function
    criterion = nn.NLLLoss()
    # Train classifier parameters
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    # 
    model.to(device)
    return model, criterion, optimizer, input_size, output_size


################################################################
# Train and Validate Image Classifier
################################################################
def image_classifier(model, criterion, optimizer, trainloader, validloader, epochs, device):
    
    # Set variables
    steps = 0
    running_loss = 0
    print_every = 10
    
    # Train model for specified number of epochs
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Accuracy calculation
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()

    return model

    
################################################################
# Test Image Classifier
################################################################
def classifier_testing(model, criterion, testloader, device):

    # Set variables
    test_loss = 0
    accuracy = 0
    
    # Test model
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Accuracy calculation
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    test_accuracy = accuracy/len(testloader)
    return test_accuracy
    

################################################################
# Save Checkpoint
################################################################
def checkpoint_saving(model, optimizer, architecture, epochs, learning_rate, input_size, output_size, hidden_layer, train_datasets):
    
    # Save label mapping of classes to indices
    model.class_to_idx = train_datasets.class_to_idx

    checkpoint = {'architecture': architecture,
                  'epochs': epochs,
                  'learning_rate': learning_rate,
                  'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layer': hidden_layer,
                  'label_mapping': model.class_to_idx,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, 'checkpoint_model.pth')


################################################################
# Main function
################################################################
def main():
    
    #
    args = argument_parser()
    
    #
    architecture = args.architecture
    hidden_layer = args.hidden_layer
    output_layer = args.output_layer
    learning_rate = args.learning_rate
    epochs = args.epochs
    gpu = args.gpu
    #data_dir = 'flowers'
    data_dir = args.data_directory
    
    # Display the various parameters
    print('architecture: ', architecture)
    print('hidden layer: ', hidden_layer)
    print('output layer: ', output_layer)
    print('learning rate: ', learning_rate)
    print('epochs: ', epochs)
    print('gpu enabled: ', gpu)
    print('data directory: ', data_dir)
    
    # Train, validation and test directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Data transformation
    trainloader, testloader, validloader, train_datasets = data_transform_load(train_dir, test_dir, valid_dir)
    
    # Select GPU or CPU
    device = gpu_selection(gpu)
    
    # Select pre-trained model and build classifier
    if not ((architecture[:-2] == 'vgg') or (architecture[:-3] == 'densenet')):
        print('Please select models among vgg and densenet architectures. Help: https://pytorch.org/docs/stable/torchvision/models.html')
    else:
        model, criterion, optimizer, input_size, output_size = build_model(architecture, learning_rate, hidden_layer, device, output_layer)
        #print(model)

        # Train and validate model
        model = image_classifier(model, criterion, optimizer, trainloader, validloader, epochs, device)

        # Test model
        test_accuracy = classifier_testing(model, criterion, testloader, device)    
        print(f"Test accuracy: {test_accuracy:.3f}")

        # Save checkpoint
        checkpoint_saving(model, optimizer, architecture, epochs, learning_rate, input_size, output_size, hidden_layer, train_datasets)


################################################################
# Run main function
################################################################
if __name__ == '__main__':
    main()




##### Sources #####

# Calling a method of a function using its str name
# https://stackoverflow.com/questions/3061/calling-a-function-of-a-module-by-using-its-name-a-string