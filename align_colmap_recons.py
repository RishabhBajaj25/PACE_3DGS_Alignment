import pycolmap
import cv2
import numpy as np
import os

# Load the reconstruction
reconstruction1 = pycolmap.Reconstruction(
    "/media/rishabh/SSD_1/Data/lab_videos_reg/2_20250324_120731_frames_10_fps/dense/1/sparse")

print(reconstruction1.summary())


# Load the reconstruction
reconstruction2 = pycolmap.Reconstruction(
    "/media/rishabh/SSD_1/Data/lab_videos_reg/1_20250324_120708_frames_10_fps/dense/0/sparse")

print(reconstruction2.summary())

rec2_from_rec1 = pycolmap.align_reconstructions_via_reprojections(reconstruction1, reconstruction2)
reconstruction1.transform(rec2_from_rec1)
print(rec2_from_rec1.scale, rec2_from_rec1.rotation, rec2_from_rec1.translation)