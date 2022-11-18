from utils import *
import matplotlib.pyplot as plt
import numpy as np

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

# Loading the DeepLabv3 model
model = utils.load_model()

# Starting a webcam video capture session
video_session = cv2.VideoCapture(0)

im1, im2, ax1, ax2 = setup_plotting(video_session)

# Define a blurring value kernel size for cv2's Gaussian Blur
blur_value = (51, 51)

# Read frames from the video, make realtime predictions and display the same
while True:
    frame = utils.grab_frame(video_session)

    # Ensure there's something in the image (not completely blank)
    if np.any(frame):
        # Read the frame's width, height, channels and get the labels' predictions from utilities
        width, height, channels = frame.shape
        labels = utils.get_pred(frame, model)
        # Wherever there's empty space/no person, the label is zero 
        # Hence identify such areas and create a mask (replicate it across RGB channels)
        mask = labels == 0
        mask = np.repeat(mask[:, :, np.newaxis], channels, axis = 2)

        # Apply the Gaussian blur for background with the kernel size specified in constants above
        blur = cv2.GaussianBlur(frame, blur_value, 0)
        frame[mask] = blur[mask]
        ax1.set_title("Blurred Video")
    
        # Set the data of the two images to frame and mask values respectively
        im1.set_data(frame)
        im2.set_data(mask * 255)
        plt.pause(0.01)
        
    else:
        break

# Empty the cache and switch off the interactive mode
torch.cuda.empty_cache()
plt.ioff()