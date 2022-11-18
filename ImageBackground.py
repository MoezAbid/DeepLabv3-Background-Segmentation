import numpy as np
from utils import *
import cv2
import torch
import matplotlib.pyplot as plt

def setup_plotting(video_session):
    # Define two axes for showing the mask and the true video in realtime
    # And set the ticks to none for both the axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 8))
    ax2.set_title("Mask Detected with Instance Segmentation")

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Create two image objects to picture on top of the axes defined above
    im1 = ax1.imshow(utils.grab_frame(video_session))
    im2 = ax2.imshow(utils.grab_frame(video_session))

    # Switch on the interactive mode in matplotlib
    plt.ion()
    plt.show()

    return im1, im2, ax1, ax2

# Background image path
BG_PTH = "background.jpg"
bg_image = cv2.imread(BG_PTH)
bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)

# Loading the DeepLabv3 model
model = utils.load_model()

# Starting a webcam video capture session
video_session = cv2.VideoCapture(0)

im1, im2, ax1, ax2 = setup_plotting(video_session)

# Read frames from the video, make realtime predictions and display the same
while True:
    frame = utils.grab_frame(video_session)

    # Ensure there's something in the image (not completely blank)
    if np.any(frame):

        # Read the frame's width, height, channels and get the labels' predictions from utilities
        width, height, channels = frame.shape
        labels = utils.get_pred(frame, model)

        # Since we are detecting a person, we specify the label as 15, which corresponds to the "person" class in the PASCAL VOC dataset
        # Subsequently repeat the mask across RGB channels 
        mask = labels == 15
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis = 2)
        
        # Resize the image for the frame capture size
        bg = cv2.resize(bg_image, (height, width))
        bg[mask] = frame[mask]
        frame = bg
        ax1.set_title("Background Trasnformation")

        # Set the data of the two images to frame and mask values respectively
        im1.set_data(frame)
        im2.set_data(mask * 255)
        plt.pause(0.01)

    else:
        break

# Empty the cache and switch off the interactive mode
torch.cuda.empty_cache()
plt.ioff()