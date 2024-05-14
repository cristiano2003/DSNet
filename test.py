import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torch
import torchvision.models as models

import torch.nn as nn
import torch


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def augment_frames(frames):
    augmented_frames = []
    for frame in frames:
        resized_frame = cv2.resize(frame, (224, 224))
        # Convert from BGR to RGB
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        # Convert frame to tensor and normalize pixel values
        tensor_frame = transforms.ToTensor()(resized_frame)
        normalized_frame = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])(tensor_frame)
        augmented_frames.append(normalized_frame)

    return augmented_frames

def frames_to_matrix(frames):
    # Convert frames list to numpy array
    frames_array = np.array(frames)
    # Reshape the array to match the desired shape (B, C, H, W)
    frames_matrix = np.transpose(frames_array, (0, 3, 1, 2))
    return frames_matrix

# Read video into frames
video_path = 'custom_data/videos/St Maarten Landing.mp4'
frames = read_video(video_path)

# Augment frames
augmented_frames = augment_frames(frames)

# Convert frames to matrix
video_matrix = frames_to_matrix(augmented_frames)

# Check the shape
print("Shape of augmented video matrix:", video_matrix.shape)





def video_matrix_to_model_input(video_matrix):
   
    
    # Stack preprocessed frames along a new dimension
    model_input = torch.stack(video_matrix, dim=0)
    
    return model_input



model_input = video_matrix_to_model_input(augmented_frames).to("cuda")

# Check the shape of model input
print("Shape of model input:", model_input.shape)


model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True).to("cuda")
model.eval()
# Define a new model without the last three layers


# Remove the last three layers (avgpool, dropout, fc)
model= nn.Sequential(*(list(model.children())[:-2]))

# # Create an instance of the modified model





with torch.no_grad():
    output = model(model_input)


print("Output shape:", output.shape)


