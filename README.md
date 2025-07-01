# Gaussian Splat Alignment in Visible Space

This branch is for aligning R2Gaussian and Gaussian Splat using manual points picked in cloud compare, It generates files that utils/apply_transform.py can use to apply the transformation to the Gaussian splats. The transformation is applied to the Gaussian splats in the `gaussian_splatting_lightning` repository. The code is not yet fully functional, but it serves as a starting point for future work.
This project focuses on aligning two Gaussian splats within either the visible or X-ray domain. The current implementation supports alignment of two 3DGS datasets in the visible RGB space. This work was conducted while affiliated with the National Institute of Informatics.

The current workflow for alignment is as follows:
1. If using R2Gaussians, convert the `.pickle` files to `.ply` files using [stricter_pickle2ply.py](https://github.com/RishabhBajaj25/r2_gaussian/blob/75c129d1653ee6bfdca2ec74b9aa659e225c0019/stricter_pickle2ply.py).
2. Clean-up the R2Gaussian and the 3DGS in [Supersplat](https://github.com/RishabhBajaj25/supersplat/tree/main).
3. Convert both types of Gaussians into point clouds using [3dgs Convertor](https://github.com/RishabhBajaj25/3dgsconverter).
4. Import these point clouds to ([CloudCompare](https://github.com/CloudCompare/CloudCompare/tree/master)). 
5. Perform alignment of 2 point clouds (either from 3DGS or R2Gaussian) in CloudCompare.
6. Save the results of the alignment from the Cloud Compare console.
7. Paste the scale value from the alignment in `scale_pcd.py` and run this script.
8. Paste the TF matrix from the alignment in `global_reg.py` and run this script.
10. Run [apply_transform.py](https://github.com/RishabhBajaj25/gaussian-splatting-lightning/blob/main/utils/apply_transform.py) in the `gaussian_splatting_lightning` fork. This applies the transformation to the Gaussian splats.
5. (Optional) Run [render_stereo.py](https://github.com/RishabhBajaj25/3DGS_PoseRender/blob/main/render_stereo.py) or [render_anaglyph.py](https://github.com/RishabhBajaj25/3DGS_PoseRender/blob/main/create_anaglyph.py) to create renders of the aligned splats. The stereo render will create a stereo image, while the anaglyph render will create a red-cyan anaglyph image.

Note: Steps 7, 8 and 9 only produce the csv/txt files necessary for the alignment, they do not perform any alignment by themselves yet. All alignment is to be done manually in CloudCompare (step 5).
**Note:** The query map is always denoted with a subscript of **2**, while the closest map is denoted with a subscript of **1**.

Additional documentation
Steps for running COLMAP from video:
### 1. Extract Frames from the Video
Run `extract_img_frames.py` to extract image frames from the video.

### 2. Run COLMAP
- Define the path to the images and specify the output directory.
- Set the camera model to **pinhole**.
- Place images directly in the directory; **do not** create an `images` folder, as this may cause issues with dense reconstruction.
- Ensure the directory structure follows this format. If necessary, create a folder named `0` and move the `.bin` files into it:

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
