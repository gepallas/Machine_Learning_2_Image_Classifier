################################################################
# Import Libraries
################################################################
import torch
import argparse
from torchvision import models
from torch import nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
import json


################################################################
# Argument Parser
################################################################
def argument_parser():
    
    parser = argparse.ArgumentParser(description='Pass image filepath to predict top classes')
    
    parser.add_argument('image_filepath', type=str,
                        help='Pass your image filepath for testing as str - example: \'flowers/test/74/image_01312.jpg\'')
    
    parser.add_argument('--topk', type=int, default=5,
                        help='Select number of top classes to be predicted - default: 5')
    
    parser.add_argument('--gpu', type=bool, default=False,
                        help='CPU is selected - default GPU: False. Turn on GPU by typing: --gpu True.')
    
    parser.add_argument('--category_to_name', type=str, default='cat_to_name.json',
                        help='Pass json file as str to map the class number to the class name - default: \'cat_to_name.json\'')
    
    args = parser.parse_args()
    return args


################################################################
# Load Checkpoint
################################################################
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu') #cuda:0
    
    method_to_call = getattr(models, checkpoint['architecture'])
    model = method_to_call(pretrained=True)
    
    hidden_layer = checkpoint['hidden_layer']
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(nn.Linear(checkpoint['input_size'], hidden_layer[0]),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_layer[0], hidden_layer[1]),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_layer[1], checkpoint['output_size']),
                                     nn.LogSoftmax(dim=1))
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['label_mapping']
    
    return model


################################################################
# Load Mapping from Category Label to Category Name
################################################################
def label_mapping(category_to_name):
    
    with open(category_to_name, 'r') as f:
        cat_to_name = json.load(f)
        
    return cat_to_name


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
# Image Preprocessing
################################################################
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Resizing of image keeping aspect ratio
    im_width, im_length = image.size
    asp_ratio = im_width/im_length
    
    if asp_ratio<=0:    
        processed_image = image.resize((256, round(256*asp_ratio)))
    else:
        processed_image = image.resize((round(256*asp_ratio), 256))
        
    # Center cropping image using pixel coordinates
    left = (processed_image.size[0]-224)/2
    upper = (processed_image.size[1]-224)/2
    right = (processed_image.size[0]-224)/2+224
    lower = (processed_image.size[1]-224)/2+224
    
    processed_image = processed_image.crop((left, upper, right, lower))
    
    # Convert values between 0 and 1
    np_image = np.array(processed_image)/255
    
    # Normalization
    mean_val = [0.485, 0.456, 0.406]
    std_val = [0.229, 0.224, 0.225]
    
    np_image = (np_image - mean_val)/std_val
    
    # Re-order dimensions
    np_image = np_image.transpose((2,0,1))
    
    return np_image


################################################################
# Class Prediction
################################################################
def predict(image_path, model, topk, cat_to_name, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Open image and process it
    pil_image = Image.open(image_path)
    np_processed_image = process_image(pil_image)
    
    # Change numpy image to torch
    # Unsqueeze image to add batch size of 1 and turn it to float to avoid runtime errors
    img = torch.from_numpy(np_processed_image).unsqueeze(0).float()
    
    # Forward pass of the image to get the top k classes and probabilities
    model.eval()
    model.to(device)
    with torch.no_grad():
        log_ps = model.forward(img)
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(topk, dim=1)
        
    # Get class numbers from top k indices and store them in a list
    top_k_class = []
    for top_class_indxs in top_class[0]:
        for lbls, indxs in model.class_to_idx.items():
            if top_class_indxs == indxs:
                top_k_class.append(lbls)
    
    # Get the names for the top k classes and store the class numbers and names in a list
    top_k_class_name = []
    i = 0
    for top_k in top_k_class:
        for lbls, fl_names in cat_to_name.items():
            if top_k == lbls:
                top_k_class_name.append((top_k, fl_names, torch.Tensor.numpy(top_p[0])[i]))
                i += 1
                
    return top_k_class_name


################################################################
# Main function
################################################################
def main():
    
    # Load checkpoint
    model = load_checkpoint('checkpoint_model.pth')
    
    # Get arguments
    args = argument_parser()    
    image_filepath = args.image_filepath
    topk = args.topk
    gpu = args.gpu
    category_to_name = args.category_to_name
    
    # Mapping from category label to name
    cat_to_name = label_mapping(category_to_name)
    
    # Select CPU or GPU
    device = gpu_selection(gpu)
    
    # Load image as PIL
    pil_image = Image.open(image_filepath)
    
    # Process image
    np_processed_image = process_image(pil_image)
    
    # Predict top classes for image
    top_k_class_name = predict(image_filepath, model, topk, cat_to_name, device)
    df_class_name_prob = pd.DataFrame(top_k_class_name, columns=['class_number', 'name', 'probability'])
    
    # Display top classes
    print(f'\nPredictions for the top {topk} probabilities for the image selected:')
    print('================================================')
    print(df_class_name_prob)
    print('================================================ \n')
    
    
################################################################
# Run main function
################################################################
if __name__ == '__main__':
    main()
    