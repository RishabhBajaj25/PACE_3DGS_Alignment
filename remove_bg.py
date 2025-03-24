from rembg import remove
import cv2
import torch
import onnxruntime as ort
# import tensorrt
import torch
from datetime import datetime
import os
from tqdm import tqdm

print('onnxruntime:', ort.get_device())
print('onnxruntime_version:', ort.__version__)
print('CUDA:',torch.version.cuda)
print('Pytorch:',torch.__version__)
print('cuda is_available:','available' if(torch.cuda.is_available()) else 'unavailable')
print('device_count:',torch.cuda.device_count())
print('device_name:',torch.cuda.get_device_name())

parent_dir = '/media/rishabh/SSD_1/Data/Foot_Blue'
output_dir = os.path.join((parent_dir), 'png_output')

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all files in the parent directory
for image_name in tqdm(os.listdir(parent_dir)):
    # Check if the file is an image (optional, based on your file types)
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
        # Generate the absolute path for the input image
        input_path = os.path.join(parent_dir, image_name)

        # Generate the output path in the output directory, appending '_rembg'
        output_name = os.path.splitext(image_name)[0] + '.png'
        # os.path.splitext(image_name)[1]
        output_path = os.path.join(output_dir, output_name)
        input = cv2.imread(input_path)
        # output = remove(input)
        output = input
        cv2.imwrite(output_path, output)