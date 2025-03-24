import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import cv2
from annoy import AnnoyIndex
import csv

# Define base directory
BASE_DIR = '/home/rishabh/Data/retrieval_test_data'
DATABASE_DIR = os.path.join(BASE_DIR, 'dataset_frames')
QUERY_IMAGE_PATH = os.path.join(BASE_DIR, 'query_img.jpg')
OUTPUT_DIR = os.path.join(BASE_DIR, 'similar_images')
CSV_FILE_PATH = os.path.join(BASE_DIR, 'similar_images.csv')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load a pre-trained EfficientNet model
class EfficientNetB4(nn.Module):
    def __init__(self):
        super(EfficientNetB4, self).__init__()
        self.model = models.efficientnet_b4(pretrained=True).features
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.model(x)
        x = self.pool(x)
        return x.flatten(start_dim=1)

# Create the EfficientNetB4 model instance
model = EfficientNetB4()
model.eval()

# Define a function to preprocess an image and extract features
def extract_features(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(img)
    # Apply L2 normalization
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.numpy()

# Extract features from all images in the database and cache them
database_features = []
database_filenames = []
for filename in os.listdir(DATABASE_DIR):
    if filename.endswith(".jpg"):
        image_path = os.path.join(DATABASE_DIR, filename)
        features = extract_features(image_path)
        database_features.append(features.flatten())  # Flatten the features
        database_filenames.append(filename)

# Create an AnnoyIndex for approximate nearest neighbor search
num_trees = 200  # Adjust for accuracy vs. speed
feature_dim = len(database_features[0])
annoy_index = AnnoyIndex(feature_dim, 'angular')  # Use cosine similarity

# Add database features to the AnnoyIndex
for i, feature in enumerate(database_features):
    annoy_index.add_item(i, feature)

# Build the index
annoy_index.build(num_trees)

# Load the query image
query_features = extract_features(QUERY_IMAGE_PATH).flatten()

# Define the number of similar images to retrieve
top_n = 5

# Find the top N similar images
similar_indices = annoy_index.get_nns_by_vector(query_features, top_n)

# Prepare a CSV file to store results
with open(CSV_FILE_PATH, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Index', 'Image Name', 'Image Path', 'Similarity Score'])

    for i, index in enumerate(similar_indices):
        similar_image_filename = database_filenames[index]
        similar_image_path = os.path.join(DATABASE_DIR, similar_image_filename)
        similar_image = cv2.imread(similar_image_path)

        # Calculate the similarity score (L2 distance)
        distance = np.linalg.norm(query_features - database_features[index])

        # Save the similar images to the output directory
        output_image_path = os.path.join(OUTPUT_DIR, similar_image_filename)
        cv2.imwrite(output_image_path, similar_image)

        # Write to CSV
        csv_writer.writerow([i + 1, similar_image_filename, similar_image_path, distance])

        print(f"Similar Image {i + 1}: Distance - {distance}")

cv2.waitKey(0)
cv2.destroyAllWindows()
