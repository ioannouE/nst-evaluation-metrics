import cv2
import numpy as np
from tkinter import Tk, Frame, TOP, BOTTOM, LEFT, Y, BOTH, CENTER, RIGHT, N, E, W
from tkinter import font
from tkinter.ttk import Label, Style
from tkinter import ttk
from PIL import Image, ImageTk
from skimage.metrics import structural_similarity as ssim
from content_style_error import content_style_error
from vgg import Vgg16
import torch

class ImageProcessorApp:
    def __init__(self, root, stylized_path, content_path, style_path):

        self.stylized_image = cv2.imread(stylized_path)
        self.content_image = cv2.imread(content_path)
        self.style_image = cv2.imread(style_path)

        self.root = root
        self.root.title("Image Noise and SSIM")
        self.root.geometry("1720x720")  # Increase window size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg = Vgg16(requires_grad=False).to(self.device)

        # Changing font-size of all the Label Widget 
        self.style = Style(self.root)
        self.style.configure("TLabel", font=('Arial', 25)) 

        self.setup_gui()
        self.load_images()
        # self.initial_calculations()

    def setup_gui(self):
        # Define fonts using font.Font for dynamic updates
        label_font = font.Font(family='Arial', size=80, weight='bold')
        slider_font = font.Font(family='Aroa;', size=60, weight='bold')


        # Configure the main window layout to use grid
        self.root.grid_rowconfigure(1, weight=1)
        
        for col in range(3):
            self.root.grid_columnconfigure(col, weight=1)

         # Set the minimum row height and column width
        self.root.grid_rowconfigure(1, minsize=100)  # Adjust the minsize as needed
        self.root.grid_columnconfigure(1, minsize=200)  # Adjust the minsize as needed


        # Frames for the images
        self.left_frame = Frame(self.root)
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        self.center_frame = Frame(self.root)
        self.center_frame.grid(row=0, column=1, sticky="nsew")

        self.right_frame = Frame(self.root)
        self.right_frame.grid(row=0, column=2, sticky="nsew")

        # Image labels
        self.ref_image_label = Label(self.left_frame)
        self.ref_image_label.pack(expand=True)

        self.image_label = Label(self.center_frame)
        self.image_label.pack(expand=True)

        self.style_image_label = Label(self.right_frame)
        self.style_image_label.pack(expand=True)

        # Noise slider
        self.noise_slider = ttk.Scale(self.center_frame, from_=0, to=30, orient="horizontal", length=400,
                                    style="Custom.Horizontal.TScale", command=self.update_image)
        self.noise_slider.pack(pady=(20, 0))  # Add padding above the slider for spacing

        # Noise level label below the slider
        self.noise_label = Label(self.center_frame, text="Noise Level: 0")
        self.noise_label.pack(pady=(5, 20))  # Add padding above and below the label

        # Metric labels with larger font, positioned on the left below the image frames
        self.ssim_label = Label(self.root, text="SSIM: ")
        self.ssim_label.grid(row=1, column=1, sticky="w", padx=100, pady=(10, 0))

        self.content_error_label = Label(self.root, text="Content Error: ", font=("Helvetica", 40))
        self.content_error_label.grid(row=1, column=1, sticky="w", padx=100, pady=(35, 0))

        self.style_error_label = Label(self.root, text="Style Error: ", font=("Helvetica", 40))
        self.style_error_label.grid(row=1, column=1, sticky="w", padx=100, pady=(60, 0))

        self.ssim_label.config(font=label_font)
        self.content_error_label.config(font=label_font)
        self.style_error_label.config(font=label_font)
        self.noise_label.config(font=slider_font)

        

    def load_images(self):
        
        self.stylized_image = cv2.resize(self.stylized_image, (512, 512))

        # Load and process the content image
        self.content_image = cv2.resize(self.content_image, (512, 512))
        self.content_gray = cv2.cvtColor(self.content_image, cv2.COLOR_BGR2GRAY)

        # Load and process the style image
        self.style_image = cv2.resize(self.style_image, (512, 512))

        # Convert images to PIL format for displaying in Tkinter
        self.current_image = Image.fromarray(cv2.cvtColor(self.stylized_image, cv2.COLOR_BGR2RGB))
        self.con_img = Image.fromarray(cv2.cvtColor(cv2.resize(self.content_image, (400,400)), cv2.COLOR_BGR2RGB))
        self.style_img = Image.fromarray(cv2.cvtColor(cv2.resize(self.style_image, (400,400)), cv2.COLOR_BGR2RGB))

        # Update Tkinter labels
        self.tk_image = ImageTk.PhotoImage(image=self.current_image)
        self.image_label.config(image=self.tk_image)

        self.tk_con_img = ImageTk.PhotoImage(image=self.con_img)
        self.ref_image_label.config(image=self.tk_con_img)

        self.tk_style_img = ImageTk.PhotoImage(image=self.style_img)
        self.style_image_label.config(image=self.tk_style_img)

        # Keep references to avoid garbage collection
        self.image_label.image = self.tk_image
        self.ref_image_label.image = self.tk_con_img
        self.style_image_label.image = self.tk_style_img


    def initial_calculations(self):
        # Initial noise level (assuming no noise initially)
        self.noise_level = 0
        self.update_image_metrics(self.noise_level, initial=True)

    def add_noise(self, image, noise_level):
        if noise_level <= 30 and noise_level > 1:
            mean = 0
            sigma = noise_level * 10
            gauss = np.random.normal(mean, sigma, image.shape).reshape(image.shape)
            noisy = image + gauss
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        else:
            noisy = image
        return noisy

    def update_image_metrics(self, noise_level, initial=False):
        if initial:
            self.noisy_image = self.stylized_image.copy()  # Use original image for initial calculation
        else:
            self.noisy_image = self.add_noise(self.stylized_image, noise_level)

        # Update the image on the GUI
        self.current_image = Image.fromarray(cv2.cvtColor(self.noisy_image, cv2.COLOR_BGR2RGB))
        self.tk_image = ImageTk.PhotoImage(image=self.current_image)
        self.image_label.config(image=self.tk_image)
        self.image_label.image = self.tk_image

        # Calculate and display SSIM
        ssim_value = ssim(cv2.cvtColor(self.noisy_image, cv2.COLOR_BGR2GRAY), self.content_gray, data_range=self.noisy_image.max() - self.noisy_image.min())
        self.ssim_label.config(text=f"SSIM: {ssim_value:.4f}", font=("Helvetica", 40))

        # Calculate and display Content and Style Error
        content_err, style_err = content_style_error(self.content_image, self.noisy_image, self.style_image, self.vgg)
        self.content_error_label.config(text=f"Content Error: {content_err:.4f}")
        self.style_error_label.config(text=f"Style Error: {style_err*100:.4f}")

    def update_image(self, value):
        noise_level = self.noise_slider.get()
        self.update_image_metrics(noise_level)
        # Display Noise Level
        self.noise_label.config(text=f"Noise Level: {noise_level:.2f}")