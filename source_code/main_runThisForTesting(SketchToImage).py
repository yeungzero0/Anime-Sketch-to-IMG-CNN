import torch
import os
import numpy as np
from PIL import Image
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image
from generator import Generator
from config import DEVICE, transform_input_image, transform_ref_image, transform_target_image

class TestDataset:
    def __init__(self, sketch_dir, ref_dir, original_dir):
        self.sketch_dir = sketch_dir
        self.ref_dir = ref_dir
        self.original_dir = original_dir
        
        self.sketch_files = sorted(os.listdir(sketch_dir))
        self.ref_files = sorted(os.listdir(ref_dir))
        self.original_files = sorted(os.listdir(original_dir))
        
        self.transform_sketch = transform_input_image
        self.transform_ref = transform_ref_image
        self.transform_original = transform_target_image
        
        self.both_transform = alb.Compose(
            [
                alb.Resize(width=256, height=256),
            ],
            # additional_targets={"image_ref": "image"}
            additional_targets={"image_real": "image", "image_ref": "image"}   # image_real->real, image->semantic(sketch), image_ref->referenced image
        )

    def __len__(self):
        # return min(len(self.sketch_files), len(self.ref_files))
        return min(len(self.sketch_files), len(self.ref_files), len(self.original_files))

    def __getitem__(self, index):
        sketch_path = os.path.join(self.sketch_dir, self.sketch_files[index])
        ref_path = os.path.join(self.ref_dir, self.ref_files[index])
        original_path = os.path.join(self.original_dir, self.original_files[index])
        
        sketch_image = Image.open(sketch_path).convert("RGB")
        ref_image = Image.open(ref_path).convert("RGB")
        original_image = Image.open(original_path).convert("RGB")
        
        sketch_image = np.array(sketch_image)
        ref_image = np.array(ref_image)
        original_image = np.array(original_image)
        
        augmented = self.both_transform(image=sketch_image, image_ref=ref_image, image_real=original_image)
        sketch_image = augmented["image"]
        ref_image = augmented["image_ref"]
        original_image = augmented["image_real"]
        
        sketch_image = self.transform_sketch(image=sketch_image)["image"]
        ref_image = self.transform_ref(image=ref_image)["image"]
        original_image = self.transform_original(image=original_image)["image"]
        
        return sketch_image, ref_image, original_image, self.sketch_files[index]

def test_model(generator, test_loader, output_dir):
    generator.eval()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with torch.no_grad():
        for sketch, ref, original, filename in test_loader:
            sketch, ref, original = sketch.to(DEVICE), ref.to(DEVICE), original.to(DEVICE)
            colorized = generator(sketch, ref)
            colorized = colorized * 0.5 + 0.5  # Denormalize
            output_path = os.path.join(output_dir, f"colorized_{filename[0]}")
            save_image(colorized, output_path)
            # Optionally save inputs for reference
            save_image(sketch * 0.5 + 0.5, os.path.join(output_dir, f"sketch_{filename[0]}"))
            save_image(ref * 0.5 + 0.5, os.path.join(output_dir, f"ref_{filename[0]}"))
            save_image(original * 0.5 + 0.5, os.path.join(output_dir, f"original_{filename[0]}"))

def main():
    # Configuration
    TEST_SKETCH_DIR = "dataset/test/sketch/"
    TEST_ORIGINAL_DIR = "dataset/test/original/"
    #TEST_REF_DIR = "dataset/val/ref/"
    TEST_REF_DIR = TEST_ORIGINAL_DIR #if you have not ref img
    
    # OUTPUT_DIR = "dataset/test/output/"
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),  'test_output')
    
    # CHECKPOINT_GEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "<Gen.pth.tar>")
    CHECKPOINT_GEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Gen.pth.tar") # PLS edit "Gen.pth.tar" to your .tar files

    # Initialize model
    generator = Generator(in_channels=6, features=64).to(DEVICE)
    
    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_GEN, map_location=DEVICE)
    generator.load_state_dict(checkpoint["state_dict"])
    
    # Create test dataset and loader
    test_dataset = TestDataset(sketch_dir=TEST_SKETCH_DIR, ref_dir=TEST_REF_DIR, original_dir=TEST_ORIGINAL_DIR)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # Process one image at a time for simplicity, def = 1/4/6/8
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Run inference
    test_model(generator, test_loader, OUTPUT_DIR)
    print(f"Colorized images saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()