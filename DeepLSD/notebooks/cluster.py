from upload_hypersim import upload_images 
import process
import utils_methods as util

import os
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lsc
import torch
import h5py

import itertools
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

from deeplsd.utils.tensor import batch_to_device
from deeplsd.models.deeplsd_inference import DeepLSD


# Model config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conf = {
    'detect_lines': True,  
    'line_detection_params': {
        'merge': False,  
        'filtering': True,  
        'grad_thresh': 3,
        'grad_nfa': True, 
    }
}

# Load the model
ckpt = '../weights/deeplsd_md.tar'
ckpt = torch.load(str(ckpt), map_location='cpu')
net = DeepLSD(conf)
net.load_state_dict(ckpt['model'])
net = net.to(device).eval()


#Define what to import
frame_str = "0001"
# images
desired_images = [
    "ai_001_001",
]
#files
files_to_download = [
    f"frame.{frame_str}.color.jpg",
    f"frame.{frame_str}.depth_meters.hdf5",
    f"frame.{frame_str}.normal_world.hdf5",
    f"frame.{frame_str}.normal_bump_world.hdf5",
    f"frame.{frame_str}.position.hdf5"
]

# Download
upload_images(desired_images, files_to_download)


# Define the data directory and the images to process
base_data_dir = "data"
image_ids = desired_images


base_data_dir = "data"
image_id = "ai_001_001"
frame_str = "0001"
image_dir = os.path.join(base_data_dir, image_id)
composite_planes = util.process_image_with_plane_detection(image_dir, 
                                                           image_id, frame_str,
                                                      spatial_weight=0.5, eps=0.5, 
                                                      min_samples=50)

