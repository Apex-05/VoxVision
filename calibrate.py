import cv2
import numpy as np
import glob
import os

CHECKERBOARD = (9, 6)
IMAGE_PATH = "data/"
SAVE_FILE = "stereo_maps.npz"

print("Starting calibration...")

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
SQUARE_SIZE_METERS = 0.024

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_METERS

objpoints = []
imgpoints_left = []
imgpoints_right = []

images_left = sorted(glob.glob(os.path.join(IMAGE_PATH, 'left', '*.png')))
images_right = sorted(glob.glob(os.path.join(IMAGE_PATH, 'right', '*.png')))

if not images_left or not images_right:
    print(f"Error: No images found. Did you run capture_images.py and save to '{IMAGE_PATH}'?")
    exit()

img_shape = None

for img_l_path, img_r_path in zip(images_left, images_right):
    img_l = cv2.imread(img_l_path)
    img_r = cv2.imread(img_r_path)

    if img_l is None or img_r is None:
        print(f"Skipping unreadable image pair: {img_l_path}, {img_r_path}")
        continue

    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    if img_shape is None:
        img_shape = gray_l.shape[::-1]

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD, None)

    if ret_l and ret_r:
        objpoints.append(objp)
        corners_l2 = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        corners_r2 = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
        imgpoints_left.append(corners_l2)
        imgpoints_right.append(corners_r2)
        print(f"Found corners in pair: {os.path.basename(img_l_path)}")
    else:
        print(f"Skipping pair: corners not found in one or both images.")

if not objpoints:
    print("No valid checkerboard pairs found. Cannot calibrate.")
    exit()

print(f"\nFound {len(objpoints)} valid image pairs.")
print("Calibrating...\n")

retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, img_shape, None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, img_shape, None, None)

flags = cv2.CALIB_FIX_INTRINSIC
retStereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtxL, distL, mtxR, distR, img_shape,
    criteria=criteria,
    flags=flags
)

if not retStereo:
    print("Stereo calibration failed.")
    exit()

print("Stereo calibration successful.\n")

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, img_shape, R, T
)
mapL1, mapL2 = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, img_shape, cv2.CV_32FC1)
mapR1, mapR2 = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, img_shape, cv2.CV_32FC1)

def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints_proj, cv2.NORM_L2) / len(imgpoints_proj)
        total_error += error
    return total_error / len(objpoints)

mean_error_left = compute_reprojection_error(objpoints, imgpoints_left, rvecsL, tvecsL, mtxL, distL)
mean_error_right = compute_reprojection_error(objpoints, imgpoints_right, rvecsR, tvecsR, mtxR, distR)
mean_error_total = (mean_error_left + mean_error_right) / 2

np.savez(
    SAVE_FILE,
    mapL1=mapL1, mapL2=mapL2,
    mapR1=mapR1, mapR2=mapR2,
    Q=Q, mtxL=mtxL, distL=distL, mtxR=mtxR, distR=distR,
    R=R, T=T, E=E, F=F
)

print("Calibration maps and matrices saved to:", SAVE_FILE)
print("\nReprojection Error Summary:")
print(f"  Left Camera Mean Error:  {mean_error_left:.4f} px")
print(f"  Right Camera Mean Error: {mean_error_right:.4f} px")
print(f"  Average Mean Error:      {mean_error_total:.4f} px")

print("\nStereo Calibration Complete!")
