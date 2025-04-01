# Gaussian Splat Alignment in Visible and X-Ray Space

This project performs the alignment of two Gaussian splats in either the visible or X-ray domain. The work was conducted while affiliated with the National Institute of Informatics.

## Steps

Note: Query map is always with a subscript of 2 and the closest map is with a subscript of 1.

1. **Extract frames from the video**  
   Run `extract_img_frames.py` to extract image frames.

2. **Run COLMAP**  
   * Define the path to the images and the output path. 
   * Ensure that the camera model is set to **pinhole**. 
   * Put images directly in the directory, DONOT create an `images` folder (creates problems with dense recon).
   * Directory structure should be as follows. If needed, make a folder named `0` and cut/paste the .bin files into it.
   ```
   project/
   ├── images/
   ├── sparse/
   │   └── 0/
   │       ├── cameras.bin
   │       ├── images.bin
   │       └── points3d.bin
   ├── stereo/
   ├── db.db
   ├── frame_0000.jpg
   ├── frame_0001.jpg
   ...
   ├── frame_0042.jpg
   ├── Fused.ply
   ├── Fused.ply.vis
   ├── run-colmap-geometric.sh
   └── run-colmap-photometric.sh
   ```

    
3. **Retrieve the closest images**  
   Select a query image from **M2** (denoted as $( I_{\text{query}}^{M2} ))$ and run `image_retrieval_efficient_net_annoy.py` to find the closest images from **M1**.

4. **Project 3D points onto the closest image**  
   Run `project_2_image.py` to project $(X_1)$ onto $(I_{\text{close}}^{M1} $).

5. **Find matching points between query and closest image**
   * Run `calc_m2_X1.py` to detect points in $(I_{\text{query}}^{M2}$) that match the projected points in $( I_{\text{close}}^{M1} $).
   * First matches the feature points between *image_query* and *image_closest*. Then it checks which of these matches are close to the projected points (within 5px distance). Usies these points and their corresponding 3D points to get X for the PnP problem.

6. **Estimate scale**  
   Run `scale_pcd.py` to estimate the scaling factor for $M_2$. We need at least a pair of query images and corresponding closest images to estimate this. Since we know what the camera pose of an image is in both the maps, by using image retrieval, the ratio of the distance between the camera centers returns the scaling factor for them.
7. **Estimate transformation using PnP**
   * Run `global_reg.py` to estimate the rotation and translation vectors. (Doesn't work too well, more testing required)
8. **Estimate transformation using ICP**  
   * Run `fine_reg.py` to estimate the rotation and translation vectors using ICP. (Dependent on the threshold value, more testing required)

   
**TODO:** 

* Check this: the scaling factor between 2_b and 1_b dataset, measured manually is ~3.262 but this files gives ~1.
* ~~How to implement this transformation to the Gaussian splats?~~: Implemented under gaussian_splatting-lightning in `apply_transform.py`

