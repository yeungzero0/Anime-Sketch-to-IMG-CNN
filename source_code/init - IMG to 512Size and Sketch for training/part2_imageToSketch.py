import os
import cv2
import numpy as np
from PIL import Image

def image_to_edge_sketch(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image formats
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    
    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            try:
                # Read the image with OpenCV
                img_path = os.path.join(input_folder, filename)
                img = cv2.imread(img_path)
                
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Apply Gaussian blur to reduce noise
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                
                # Apply adaptive thresholding to enhance edges
                thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY_INV, 11, 2)
                
                # Apply Canny edge detection
                edges = cv2.Canny(blurred, 100, 200)
                
                # Combine threshold and edges for a clean sketch effect
                sketch = cv2.bitwise_or(thresh, edges)
                
                # Invert to get black lines on white background
                sketch = cv2.bitwise_not(sketch)
                
                # Convert back to RGBA using PIL for consistent output
                sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
                sketch_rgba = Image.fromarray(sketch_rgb).convert('RGBA')
                
                # Calculate scaling factor to fit within 512x512 while preserving aspect ratio
                img_width, img_height = sketch_rgba.size
                scale = min(512 / img_width, 512 / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                
                # Resize image
                sketch_rgba = sketch_rgba.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Create a new 512x512 transparent background
                new_img = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
                
                # Paste the resized sketch in the center
                offset = ((512 - new_width) // 2, (512 - new_height) // 2)
                new_img.paste(sketch_rgba, offset)
                
                # Save as PNG
                output_filename = os.path.splitext(filename)[0] + '_edge_sketch.png'
                output_path = os.path.join(output_folder, output_filename)
                new_img.save(output_path, 'PNG')
                
                #testing error
                #print(f"Processed: {filename} -> {output_filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
        else:
            print(f"Skipped: {filename} (unsupported format)")

# Example usage
input_folder = 'dataset/newIMG/to512size'  # Your input folder path
output_folder = 'dataset/newIMG/toSketch'  # Output folder for sketch images
image_to_edge_sketch(input_folder, output_folder)