import os
from PIL import Image

def resize_to_512x512(input_folder, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported image formats
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    
    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            try:
                # Open the image
                img_path = os.path.join(input_folder, filename)
                img = Image.open(img_path).convert('RGBA')
                
                # Calculate scaling factor to fit within 512x512 while preserving aspect ratio
                img_width, img_height = img.size
                scale = min(512 / img_width, 512 / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                
                # Resize image
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Create a new 512x512 white background
                new_img = Image.new('RGBA', (512, 512), (255, 255, 255, 255))
                
                # Paste the resized image in the center
                offset = ((512 - new_width) // 2, (512 - new_height) // 2)
                new_img.paste(img, offset, img)  # Use img as mask to preserve transparency if any
                
                # Save as PNG
                output_filename = os.path.splitext(filename)[0] + '.png'
                output_path = os.path.join(output_folder, output_filename)
                new_img.save(output_path, 'PNG')
                
                #testing error
                #print(f"Processed: {filename} -> {output_filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
        else:
            print(f"Skipped: {filename} (unsupported format)")

# Example usage
input_folder = 'dataset/newIMG/original'  # Your input folder path
output_folder = 'dataset/newIMG/to512size'  # Output folder for resized images
resize_to_512x512(input_folder, output_folder)