import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image
import albumentations as alb
from albumentations.pytorch import ToTensorV2

# ------------------------------
# Configuration
# ------------------------------
# Use GPU if available, otherwise fallback to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directories for training and validation datasets
TRAIN_DIR_REAL = "dataset/train/real/"
#TRAIN_DIR_REF = "dataset/train/ref/" #if you have your own ref
TRAIN_DIR_REF = TRAIN_DIR_REAL 
TRAIN_DIR_SKETCH = "dataset/train/sketch/"

VAL_DIR_REAL = "dataset/val/real/"
#VAL_DIR_REF = "dataset/val/ref/"#if you have your own ref
VAL_DIR_REF = VAL_DIR_REAL 
VAL_DIR_SKETCH = "dataset/val/sketch/"

# Parameters for traing
LEARNING_RATE = 2e-4  # Initial learning rate for the optimizer
L1_WEIGHT = 100       # Weight for L1 loss
FM_WEIGHT = 10        # Weight for feature matching loss

# processing
# BATCH_SIZE = 48           # Number of samples per batch
BATCH_SIZE = 32           # Number of samples per batch
NUM_EPOCHS = 1001          # Number of epochs to train
DISC_SCALES = [1.0, 0.5]  # Scales for multi-discriminator (full and half resolution)
RANDOM = True

# model

SCPATH = os.path.dirname(os.path.abspath(__file__))
SAVE_MODEL = True  # Whether to save model checkpoints
LOAD_MODEL = True  # Whether to load pre-trained models
CHECKPOINT_DISC = os.path.join(SCPATH,  'Disc.pth.tar')  # Discriminator checkpoint
CHECKPOINT_GEN = os.path.join(SCPATH,  'Gen.pth.tar')    # Generator checkpoint



# Image transformation pipelines
both_transform = alb.Compose(
    [
        alb.Resize(width=256, height=256),
        alb.HorizontalFlip(p=0.3),  # Apply horizontal flip to both images
        alb.Rotate(limit=15, p=0.3),  # Add slight rotation for reference diversity
        #alb.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
    ],
#     additional_targets={"image_real": "image"}  # image: semantic, image_real: real
# )

    additional_targets={"image_real": "image", "image_ref": "image"}   # image_real->real, image->semantic(sketch), image_ref->referenced image
)

transform_ref_image = alb.Compose(
    [
        alb.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

transform_input_image = alb.Compose(
    [
        alb.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

transform_target_image = alb.Compose(
    [
        alb.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

# ------------------------------
# Image of Dataset
# ------------------------------
class ImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, ref_dir, sample_size=None):  
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.ref_dir = ref_dir
        self.input_image_files = os.listdir(self.input_dir)
        self.target_image_files = os.listdir(self.target_dir)
        self.ref_image_files = os.listdir(self.ref_dir)
        if sample_size is not None:
            self.input_image_files = self.input_image_files[:sample_size]
            self.target_image_files = self.target_image_files[:sample_size]
            # Do not limit reference images; they will be sampled randomly
        assert len(self.input_image_files) == len(self.target_image_files), "Mismatched sketch and target images"
        
    def __getitem__(self, index):
        input_img_file = self.input_image_files[index]
        target_img_file = self.target_image_files[index]
        # Randomly select a reference image
        ref_img_file = random.choice(self.ref_image_files)
        
        input_img_path = os.path.join(self.input_dir, input_img_file)
        target_img_path = os.path.join(self.target_dir, target_img_file)
        ref_img_path = os.path.join(self.ref_dir, ref_img_file)
        
        input_image = Image.open(input_img_path).convert("RGB")
        target_image = Image.open(target_img_path).convert("RGB")
        ref_image = Image.open(ref_img_path).convert("RGB")
        
        input_image = np.array(input_image)
        target_image = np.array(target_image)
        ref_image = np.array(ref_image)
        
        augmented = both_transform(image=input_image, image_real=target_image, image_ref=ref_image)
        input_image = augmented["image"]
        target_image = augmented["image_real"]
        ref_image = augmented["image_ref"]
        
        input_image = transform_input_image(image=input_image)["image"]
        target_image = transform_target_image(image=target_image)["image"]
        ref_image = transform_ref_image(image=ref_image)["image"]
        
        return input_image, target_image, ref_image
    
    def __len__(self):
        return len(self.input_image_files)