import argparse
import os
import sys

from skimage import io, transform
from torch.utils.data import DataLoader
import torch
torch.cuda.empty_cache()

import gc
gc.collect()

import re
import numpy as np
from scipy import misc
from PIL import Image
from torch.autograd import Variable
import glob
from torchvision import transforms
import matplotlib as plt
import cv2

import lpips

from vgg import Vgg16
import utils

# https://github.com/safwankdb/ReCoNet-PyTorch/blob/master/testwarp.py

def get_subdirectories(directory_path):
    subdirectories = []
    for entry in os.scandir(directory_path):
        if entry.is_dir():
            subdirectories.append(entry.name)
    return subdirectories



def main():
    
    parser = argparse.ArgumentParser(description='parser for evaluating a model')
    parser.add_argument("--content-image", type=str, required=True,
                        help="folder that contains the images")
    parser.add_argument("--style-image", type=str, required=True,
                                  help="the style image")
    parser.add_argument("--stylized-image", type=str, required=True,
                                  help="the style image")
    parser.add_argument("--cuda", type=int, default=1, required=False,
                                  help="use cuda")
    parser.add_argument("--image-size", type=int, default=360,
                                  help="the image size")
    
    
    args = parser.parse_args()

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

     # set up midas
    device = torch.device("cuda" if args.cuda else "cpu")
    print("Device: ", torch.cuda.get_device_name(0))
    print("Running on ", device)

    mse_loss = torch.nn.MSELoss()
    lpips_sim = lpips.LPIPS(net='squeeze').to(device)
    vgg = Vgg16(requires_grad=False).to(device)

    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(args.style_image)
    style = style_transform(style).to(device)
    style = style.repeat(1, 1, 1, 1).to(device)
    
    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    
    sum_content = 0.
    sum_style = 0.
    #############################################################################
    
    content_image = Image.open(args.content_image).convert('RGB')
    #if isinstance(image, Image.Image):
    
    content_image = transforms.Resize((args.image_size,args.image_size))(content_image)
    content_image = transforms.ToTensor()(content_image)
    content_image = content_image.unsqueeze(0).to(device)



    stylized_image = Image.open(args.stylized_image).convert('RGB')
    #if isinstance(image, Image.Image):
    
    stylized_image = transforms.Resize((args.image_size,args.image_size))(stylized_image)
    stylized_image = transforms.ToTensor()(stylized_image)
    stylized_image = stylized_image.unsqueeze(0).to(device)


    content_error = 0.
    style_error = 0.
        
    features_org = vgg(content_image)
    features_stylized = vgg(stylized_image)
    content_error += mse_loss(features_stylized.relu2_2, features_org.relu2_2)

    style_loss = 0.
    for ft_y, gm_s in zip(features_stylized, gram_style):
        gm_y = utils.gram_matrix(ft_y)
        style_loss += mse_loss(gm_y, gm_s)
    style_error += style_loss


def content_style_error(content_image, stylized_image, style_image, vgg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = 512

    mse_loss = torch.nn.MSELoss()
    # lpips_sim = lpips.LPIPS(net='squeeze').to(device)

    style_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x.mul(255))
    ])
    # style = utils.load_image(style_image)
    style = style_transform(style_image).to(device)
    style = style.repeat(1, 1, 1, 1).to(device)
    
    # features_style = vgg(utils.normalize_batch(style))
    features_style = vgg(style)
    gram_style = [utils.gram_matrix(y) for y in features_style]


    # content_image = Image.open(content_image).convert('RGB')   
    content_image = transforms.ToPILImage()(content_image)
    content_image = transforms.Resize((image_size,image_size))(content_image)
    content_image = transforms.ToTensor()(content_image)
    content_image = content_image.unsqueeze(0).to(device)



    # stylized_image = Image.open(stylized_image).convert('RGB')    
    stylized_image = transforms.ToPILImage()(stylized_image)
    stylized_image = transforms.Resize((image_size,image_size))(stylized_image)
    stylized_image = transforms.ToTensor()(stylized_image)
    stylized_image = stylized_image.unsqueeze(0).to(device)


    content_error = 0.
    style_error = 0.
        
    features_org = vgg(content_image)
    features_stylized = vgg(stylized_image)
    content_error += mse_loss(features_stylized.relu2_2, features_org.relu2_2)

    style_loss = 0.
    for ft_y, gm_s in zip(features_stylized, gram_style):
        gm_y = utils.gram_matrix(ft_y)
        style_loss += mse_loss(gm_y, gm_s)
    style_error += style_loss

    return content_error, style_error

    
if __name__ == "__main__":
    main()