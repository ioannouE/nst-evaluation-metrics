import argparse
import cv2
import numpy as np
from tkinter import *
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from skimage.metrics import structural_similarity as ssim
from content_style_error import content_style_error
from vgg import Vgg16
import torch
from image_processor_app import ImageProcessorApp


def parse_arguments():
    parser = argparse.ArgumentParser(description='Image Processor Application')
    parser.add_argument('--stylized', type=str, required=True, help='Path to the original image')
    parser.add_argument('--content', type=str, required=True, help='Path to the content image')
    parser.add_argument('--style', type=str, required=True, help='Path to the style image')
    return parser.parse_args()


# Running the Application
if __name__ == "__main__":
    args = parse_arguments()

    root = Tk()
    app = ImageProcessorApp(root, args.stylized, args.content, args.style)
    root.mainloop()