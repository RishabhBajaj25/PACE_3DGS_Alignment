import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import cv2
from annoy import AnnoyIndex
import csv

# File to retrieve similar images from a database

# Define base directory paths
BASE_DIR = '/media/rishabh/SSD_1/Data/lab_videos_reg'
DATABASE_DIR = os.path.join(BASE_DIR, '1_20250324_120708_frames_10_fps/images')  # Directory containing database images
QUERY_IMAGE_PATH = os.path.join(BASE_DIR, '2_20250324_120731_frames_10_fps/images/frame_0000.jpg')  # Query image
OUTPUT_DIR = os.path.join(BASE_DIR, 'similar_images_from_1_20250324_120708_frames_10_fps')  # Output directory for similar images
CSV_FILE_PATH = os.path.join(BASE_DIR, 'similar_images_from_1_20250324_120708_frames_10_fps.csv')  # CSV output file

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load a pre-trained EfficientNet model for feature extraction
class EfficientNetB4(nn.Module):
    def __init__(self):
        super(EfficientNetB4, self).__init__()
        self.model = models.efficientnet_b4(pretrained=True).features  # Load EfficientNetB4 without classification head
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling layer

    def forward(self, x):
        x = self.model(x)
        x = self.pool(x)
        return x.flatten(start_dim=1)  # Flatten the output to a 1D feature vector

# Initialize the model and set it to evaluation mode
model = EfficientNetB4()
model.eval()

# Define a function to preprocess an image and extract feature descriptors
def extract_features(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
    ])
    img = Image.open(image_path).convert('RGB')  # Load and convert to RGB
    img = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(img)  # Extract features
    features = features / features.norm(p=2, dim=-1, keepdim=True)  # Apply L2 normalization
    return features.numpy()

# Extract features for all images in the database

database_features = []  # List to store feature vectors
database_filenames = []  # List to store image filenames

for filename in os.listdir(DATABASE_DIR):
    if filename.endswith(".jpg"):
        image_path = os.path.join(DATABASE_DIR, filename)
        features = extract_features(image_path)
        database_features.append(features.flatten())  # Flatten the feature vector
        database_filenames.append(filename)

# Create an Annoy index for fast approximate nearest neighbor search
num_trees = 200  # More trees = better accuracy but slower indexing
feature_dim = len(database_features[0])  # Feature vector dimension
annoy_index = AnnoyIndex(feature_dim, 'angular')  # Using cosine similarity

# Add database features to the Annoy index
for i, feature in enumerate(database_features):
    annoy_index.add_item(i, feature)

# Build the Annoy index
annoy_index.build(num_trees)

# Extract features from the query image
query_features = extract_features(QUERY_IMAGE_PATH).flatten()

# Define the number of similar images to retrieve
top_n = 5

# Retrieve the top-N most similar images from the database
similar_indices = annoy_index.get_nns_by_vector(query_features, top_n)

# Prepare a CSV file to store the results
with open(CSV_FILE_PATH, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Index', 'Query Image Path', 'Image Name', 'Image Path', 'Similarity Score'])

    for i, index in enumerate(similar_indices):
        similar_image_filename = database_filenames[index]
        similar_image_path = os.path.join(DATABASE_DIR, similar_image_filename)
        similar_image = cv2.imread(similar_image_path)

        # Compute similarity score (L2 distance)
        distance = np.linalg.norm(query_features - database_features[index])

        # Save the similar images to the output directory
        output_image_path = os.path.join(OUTPUT_DIR, similar_image_filename)
        cv2.imwrite(output_image_path, similar_image)

        # Write results to CSV
        csv_writer.writerow([i + 1, QUERY_IMAGE_PATH, similar_image_filename, similar_image_path, distance])

        print(f"Similar Image {i + 1}: Distance - {distance}")

cv2.waitKey(0)
cv2.destroyAllWindows()