# Gaussian Splat Alignment in Visible and X-Ray Space

This project performs the alignment of two Gaussian splats in either the visible or X-ray domain. The work was conducted while affiliated with the National Institute of Informatics.

## Steps

1. **Extract frames from the video**  
   Run `extract_img_frames.py` to extract image frames.

2. **Run COLMAP**  
   Define the path to the images and the output path. Ensure that the camera model is set to **pinhole**. Put images directly in the directory, DONOT create an `images` folder (creates problems with dense recon).

3. **Retrieve the closest images**  
   Select a query image from **M2** (denoted as $( I_{\text{query}}^{M2} ))$ and run `image_retrieval_efficient_net_annoy.py` to find the closest images from **M1**.

4. **Project 3D points onto the closest image**  
   Run `project_2_image.py` to project $(X_1)$ onto $(I_{\text{close}}^{M1} $).

5. **Find matching points between query and closest image**  
   Run `closest_feature_matcher.py` to detect points in $(I_{\text{query}}^{M2}$) that match the projected points in $( I_{\text{close}}^{M1} $).

**TODO:** 

* Triangulate the common points in $( I_{\text{close}} $) and $( I_{\text{query}} $) to get their 3D positions, $( X_{\text{common}} $).
* **Estimate pose using PnP**: Use `solvePnPRANSAC(X_common, x_common_M2, K2, distCoeffs)` to estimate the rotation and translation vectors.
* **Determine scale from camera positions**: Develop a method to compute scales using camera positions.

