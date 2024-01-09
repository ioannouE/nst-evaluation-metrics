import cv2
import numpy as np
from tkinter import *
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from skimage.metrics import structural_similarity as ssim
from content_style_error import content_style_error
from vgg import Vgg16

def center_crop(img, crop_size):
    height, width = img.shape[:2]
    crop_height, crop_width = crop_size

    # Calculate the top-left corner of the crop area
    start_x = width // 2 - crop_width // 2
    start_y = height // 2 - crop_height // 2

    # Crop the image
    cropped_img = img[start_y:start_y + crop_height, start_x:start_x + crop_width]
    return cropped_img


def add_noise(image, noise_level):
    mean = 0
    sigma = noise_level * 10
    gauss = np.random.normal(mean, sigma, image.shape).reshape(image.shape)
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def initial_calculations():
    # Initial noise level (assuming no noise initially)
    noise_level = 0
    update_image_metrics(noise_level, initial=True)

# def update_image(value):
#     global current_image
#     noise_level = noise_slider.get()
#     noisy_image = add_noise(original_image, noise_level)
#     current_image = Image.fromarray(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
#     tk_image = ImageTk.PhotoImage(image=current_image)
#     image_label.config(image=tk_image)
#     image_label.image = tk_image

#     # Calculate and display SSIM
#     ssim_value = ssim(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY), reference_gray, data_range=noisy_image.max() - noisy_image.min())
#     ssim_label.config(text=f"SSIM: {ssim_value:.4f}")

def update_image_metrics(noise_level, initial=False):
    global current_image, noisy_image

    if initial:
        noisy_image = original_image.copy()  # Use original image for initial calculation
    else:
        noisy_image = add_noise(original_image, noise_level)

    # Update the image on the GUI
    current_image = Image.fromarray(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
    tk_image = ImageTk.PhotoImage(image=current_image)
    image_label.config(image=tk_image)
    image_label.image = tk_image

    # Calculate and display SSIM
    ssim_value = ssim(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY), content_gray, data_range=noisy_image.max() - noisy_image.min())
    ssim_label.config(text=f"SSIM: {ssim_value:.4f}")

    # Calculate and display Content and Style Error
    content_err, style_err = content_style_error(content_image, noisy_image, style_image, vgg)
    content_error_label.config(text=f"Content Error: {content_err:.4f}")
    style_error_label.config(text=f"Style Error: {style_err*100:.4f}")

    # Display Noise Level
    noise_label.config(text=f"Noise Level: {noise_level:.2f}")

def update_image(value):
    # Get the current value of the slider as the noise level
    noise_level = noise_slider.get()
    update_image_metrics(noise_level)

# def update_image(value):
    global current_image
    global noisy_image
    global style_image
    global content_image
    current_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    noise_level = noise_slider.get()
    max_noise_level = 30  # Assuming the slider max value is 20

    if noise_level < max_noise_level and noise_level>1:
        noisy_image = add_noise(original_image, noise_level)
    # else:
    # #     # If the slider is at the maximum, display the image without additional noise
    # #     # Change this to apply maximum noise if desired
    #     noisy_image = add_noise(original_image, max_noise_level)
    #     # noisy_image = add_noise(noisy_image, 0)


    current_image = Image.fromarray(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
    tk_image = ImageTk.PhotoImage(image=current_image)
    image_label.config(image=tk_image)
    image_label.image = tk_image

    # Calculate and display SSIM
    ssim_value = ssim(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY), content_gray, data_range=noisy_image.max() - noisy_image.min())
    ssim_label.config(text=f"SSIM: {ssim_value:.4f}")

    # Calculate and display Content Error
    content_err, style_err = content_style_error(content_image, noisy_image, style_image, vgg)
    content_error_label.config(text=f"Content Error: {content_err:.4f}")
    style_error_label.config(text=f"Style Error: {style_err*100:.4f}")


vgg = Vgg16(requires_grad=False).to("cpu")

# Load images
# original_image = cv2.imread('images/cornell_stylized_starry_night.jpg')
original_image = cv2.imread('images/cornell.jpg')
original_image = cv2.resize(original_image, (512, 512))

content_image = cv2.imread('images/cornell.jpg')
# content_image = center_crop(content_image, (512,512))
content_image = cv2.resize(content_image, (512, 512))
content_gray = cv2.cvtColor(content_image, cv2.COLOR_BGR2GRAY)

style_image = cv2.imread('images/starry_night.jpg')
# style_image = center_crop(style_image, (512,512))
# style_image_resized = cv2.resize(style_image, (512, 512))

# Setup GUI
root = Tk()
root.title("Image Noise and SSIM")

# Frame for the main image and SSIM label
main_frame = Frame(root)
main_frame.pack(side=TOP, fill=BOTH, expand=True)

# Frame for the reference image
ref_frame = Frame(main_frame)
ref_frame.pack(side=LEFT, fill=Y)

# Frame for the style image
style_frame = Frame(ref_frame)
style_frame.pack(side=TOP, fill=Y, anchor=N)

current_image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
tk_image = ImageTk.PhotoImage(image=current_image)

image_label = Label(main_frame, image=tk_image)
image_label.pack(side=LEFT)

ssim_label = Label(main_frame, text="SSIM: ", justify=CENTER)
ssim_label.pack(side=BOTTOM)
content_error_label = Label(main_frame, text="Content Error: ", justify=RIGHT)
content_error_label.pack(side=BOTTOM)
style_error_label = Label(main_frame, text="Style Error: ", justify=RIGHT)
style_error_label.pack(side=BOTTOM)


# Display the smaller reference image
con_img = content_image = cv2.resize(content_image, (100, 100))
con_img = Image.fromarray(cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB))
tk_ref_img = ImageTk.PhotoImage(image=con_img)

style_img = style_image = cv2.resize(style_image, (100, 100))
style_img = Image.fromarray(cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB))
tk_style_img = ImageTk.PhotoImage(image=style_img)

ref_image_label = Label(ref_frame, image=tk_ref_img)
ref_image_label.image = tk_ref_img  # Keep a reference to avoid garbage collection
ref_image_label.pack()

style_image_label = Label(style_frame, image=tk_style_img)
style_image_label.image = tk_style_img  # Keep a reference to avoid garbage collection
style_image_label.pack()

# Label for displaying noise level
noise_label = Label(root, text="Noise Level: 0")
noise_label.pack(side=BOTTOM, pady=5)

# Initial calculations and display
initial_calculations()

# Slider at the bottom
# noise_slider = Scale(root, from_=0, to=20, orient=HORIZONTAL, length=300, sliderlength=20, command=update_image)
# noise_slider.pack(side=BOTTOM, pady=10)

# Set a theme for ttk
style = ttk.Style()
style.theme_use('clam')  # You can try other themes like 'alt', 'default', 'classic', 'vista'

# Configure the style of the horizontal ttk Scale widget
style.configure("Custom.Horizontal.TScale", background="white", foreground="#fff",
                troughcolor="#ccc", bordercolor="blue", lightcolor="#333", darkcolor="#333",
                sliderlength=30, width=20)

# Create and pack the ttk Scale slider
noise_slider = ttk.Scale(root, from_=0, to=30, orient="horizontal", length=300, style="Custom.Horizontal.TScale", command=update_image)
noise_slider.pack(side=tk.BOTTOM, pady=10)

root.mainloop()