# 3D Scene Perception Using Low-Cost Stereo Cameras

## Project Overview
This project demonstrates 3D scene perception and object segmentation using low-cost stereo cameras. It leverages stereo camera calibration, disparity map computation, clustering, foreground-background segmentation, and voxel grid visualization for effective spatial understanding and segmentation in complex environments.

## Folder Structure
- `capture_images.py` — Script to capture stereo images needed for calibration.
- `calibrate.py` — Camera calibration script to generate intrinsic/extrinsic parameters and rectification maps, saved as `stereomaps.npz`.
- `stereomaps.npz` — Stereo calibration results containing rectification maps, camera matrices, distortion coefficients, and reprojection matrix.
- `live_depth.py` — Real-time depth estimation from stereo cameras with disparity map generation.
- `kmeans_clustering.py` — Uses k-means clustering on depth maps for segmentation.
- `fg_bg_segmentation.py` — Foreground-background segmentation using disparity and morphological filters.
- `voxel_grid.py` — Creates 3D voxel grid visualization from disparity maps.

## Step-by-Step Workflow

### 1. Image Capture and Calibration
- Use `capture_images.py` to capture paired stereo images of a calibration checkerboard pattern.
- Run `calibrate.py` which detects checkerboard corners, performs stereo camera calibration, and stereo rectification.
- Outputs calibration and rectification maps in `stereomaps.npz` for use in later steps.

### 2. Real-Time Stereo Depth Estimation
- Use `live_depth.py` (or `live_depth_new_trajectory.py`) to load calibration maps and capture live video feeds from stereo cameras.
- Frames are rectified using precomputed stereo rectification maps.
- Disparity maps are computed using StereoSGBM algorithm with optional filtering.
- Parameters like disparity number, block size, and thresholds can be tuned via GUI trackbars.

### 3. Segmentation Techniques
- `kmeans_clustering.py` applies k-means clustering on disparity maps for scene segmentation.
- `fg_bg_segmentation.py` performs foreground-background segmentation using disparity thresholds and morphological operations.

### 4. 3D Visualization with Voxel Grids
- `voxel_grid.py` constructs voxel grids from disparity-based 3D point clouds for spatial visualization of the scene.
- Users can tune voxel size and depth thresholds interactively.

## Additional Notes
- Camera identifiers (`CAMLEFTID`, `CAMRIGHTID`) are set at the start of each script and may need adjustment based on user system.
- The project uses OpenCV for image processing and Open3D for 3D visualizations.
- Real-time processing performance depends on hardware and camera frame rates.
- Calibration quality is critical for accurate depth estimation and segmentation.

## Dependencies
- Python 3.x
- OpenCV 
- numpy
- open3d
