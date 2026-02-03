# Gaussian Splat Alignment in Visible Space

This branch is for aligning R2Gaussian and Gaussian Splat using manual points picked in CloudCompare. It generates files that `utils/apply_transform.py` in the [`gaussian_splatting_lightning`](https://github.com/RishabhBajaj25/gaussian-splatting-lightning) repo can use to apply the transformation to the Gaussian splats. The code is not yet fully functional but serves as a starting point for future work.

This project focuses on aligning two Gaussian splats within either the visible or X-ray domain. The current implementation supports alignment of two 3DGS datasets in the visible RGB space. This work was conducted while affiliated with the National Institute of Informatics.

---

## Dependencies

This project uses the following tools:

- [`CloudCompare`](https://github.com/CloudCompare/CloudCompare) (desktop tool for manual alignment)
- [`gaussian_splatting_lightning`](https://github.com/RishabhBajaj25/gaussian-splatting-lightning) (for transforming and rendering splats)
- [`r2_gaussian`](https://github.com/RishabhBajaj25/r2_gaussian) (for converting .pickle to .ply)
- [`3dgsconverter`](https://github.com/RishabhBajaj25/3dgsconverter)
- [`supersplat`](https://github.com/RishabhBajaj25/supersplat) (for visualization and debugging)

---

## Workflow for Alignment

1. **Convert R2Gaussians to PLY:**
   If using R2Gaussians, convert the `.pickle` files to `.ply` files using [`stricter_pickle2ply.py`](https://github.com/RishabhBajaj25/r2_gaussian/blob/75c129d1653ee6bfdca2ec74b9aa659e225c0019/stricter_pickle2ply.py).

2. **Clean-up in Supersplat:**
   Clean up the R2Gaussian and 3DGS using [Supersplat](https://github.com/RishabhBajaj25/supersplat/tree/main).

3. **Convert to Point Clouds:**
   Convert both types of Gaussians into point clouds using the [3dgs Convertor](https://github.com/RishabhBajaj25/3dgsconverter).

4. **Import into CloudCompare:**
   Import these point clouds into [CloudCompare](https://github.com/CloudCompare/CloudCompare/tree/master).

5. **Manual Alignment:**
   Perform alignment of the 2 point clouds (either from 3DGS or R2Gaussian) manually in CloudCompare [click here for more information on alignment](https://www.cloudcompare.org/doc/wiki/index.php/Alignment_and_Registration).

6. **Save Alignment Results:**
   Save the results of the alignment from the CloudCompare console.

7. **Scale Point Cloud:**
   Paste the scale value from the alignment in `scale_pcd.py` and run this script.

8. **Apply Transformation Matrix:**
   Paste the transformation matrix from the alignment in `global_reg.py` and run this script.

9. **Perform ICP:**
   Run `fine_reg.py`.

10. **Apply Transformation to Gaussians:**
   Run [`apply_transform.py`](https://github.com/RishabhBajaj25/gaussian-splatting-lightning/blob/main/utils/apply_transform.py) in the `gaussian_splatting_lightning` fork. This applies the transformation to the Gaussian splats.

11. **(Optional) Create Renders:**
    - Run [`render_stereo.py`](https://github.com/RishabhBajaj25/3DGS_PoseRender/blob/main/render_stereo.py) to create a stereo image.
    - Run [`create_anaglyph.py`](https://github.com/RishabhBajaj25/3DGS_PoseRender/blob/main/create_anaglyph.py) to create a red-cyan anaglyph image.

> **Note:** Steps 7, 8, and 9 only produce the CSV/TXT files necessary for the alignment. They do not perform alignment by themselves. All alignment is performed manually in CloudCompare (Step 5).

> **Note:** The query map is always denoted with a subscript of **2**, while the closest map is denoted with a subscript of **1**.

---

## Additional Documentation

### Running COLMAP from Video

#### 1. Extract Frames from the Video
Run `extract_img_frames.py` to extract image frames from the video.

#### 2. (optional) Extract only sharp frames from the dataset
After image frames have been extracted from video, run `preprocess.py` [here](https://github.com/RishabhBajaj25/gaussian-splatting/blob/main/preprocess.py) extract only sharp frames from the dataset. This is useful for reducing the number of images and focusing on the most relevant ones.

#### 3. Run COLMAP
- Define the path to the images and specify the output directory.
- Set the camera model to **pinhole**.
- Place images directly in the directory. **Do not** create an `images` folder, as this may cause issues with dense reconstruction.
- Ensure the directory structure follows this format:

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
