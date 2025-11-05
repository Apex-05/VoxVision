import cv2
import numpy as np
import time
import os

CAM_LEFT_ID  = 2
CAM_RIGHT_ID = 1
CALIB_FILE   = "stereo_maps.npz"
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
MIN_VALID_POINTS = 50
OUT_DIR = "seg_output"
os.makedirs(OUT_DIR, exist_ok=True)

try:
    with np.load(CALIB_FILE) as data:
        mapL1, mapL2 = data['mapL1'], data['mapL2']
        mapR1, mapR2 = data['mapR1'], data['mapR2']
        Q = data['Q']
    print(f"Loaded calibration from {CALIB_FILE}")
except Exception as e:
    raise SystemExit(f"Could not load '{CALIB_FILE}': {e}")

capL = cv2.VideoCapture(CAM_LEFT_ID)
capR = cv2.VideoCapture(CAM_RIGHT_ID)
if not (capL.isOpened() and capR.isOpened()):
    raise SystemExit("Could not open both cameras — check CAM IDs.")

for c in (capL, capR):
    c.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    c.set(cv2.CAP_PROP_FPS, 30)

time.sleep(1.0)

cv2.namedWindow("Tuning", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tuning", 640, 320)
def nothing(x): pass

cv2.createTrackbar("numDisp(x16)", "Tuning", 6, 12, nothing)
cv2.createTrackbar("blockSize", "Tuning", 5, 15, nothing)
cv2.createTrackbar("uniqueness", "Tuning", 10, 30, nothing)
cv2.createTrackbar("PreFilterCap", "Tuning", 31, 63, nothing)
cv2.createTrackbar("disp_thresh", "Tuning", 10, 200, nothing)
cv2.createTrackbar("morph_k", "Tuning", 5, 25, nothing)
cv2.createTrackbar("alpha_pos(0-100)", "Tuning", 80, 100, nothing)
cv2.createTrackbar("alpha_depth(0-100)", "Tuning", 85, 100, nothing)

print("\nPress 's' to save a snapshot, 'q' to quit.\n")

prev_time = time.time()
smoothed_pos = None
smoothed_Z = None

def make_sgbm(numDisp, blockSize, uniqueness, preFilterCap=31):
    if blockSize % 2 == 0:
        blockSize += 1
    P1 = 8 * 3 * blockSize**2
    P2 = 32 * 3 * blockSize**2
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=numDisp,
        blockSize=blockSize,
        P1=P1, P2=P2,
        disp12MaxDiff=1,
        preFilterCap=preFilterCap,
        uniquenessRatio=uniqueness,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    return sgbm

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not (retL and retR):
        print("Frame grab failed. Exiting.")
        break

    rectL = cv2.remap(frameL, mapL1, mapL2, cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, mapR1, mapR2, cv2.INTER_LINEAR)

    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    numDisp = max(16, cv2.getTrackbarPos("numDisp(x16)", "Tuning") * 16)
    blockSize = cv2.getTrackbarPos("blockSize", "Tuning")
    blockSize = blockSize + 1 if blockSize % 2 == 0 else blockSize
    uniq = cv2.getTrackbarPos("uniqueness", "Tuning")
    preCap = cv2.getTrackbarPos("PreFilterCap", "Tuning")
    disp_thresh = cv2.getTrackbarPos("disp_thresh", "Tuning")
    morph_k = max(1, cv2.getTrackbarPos("morph_k", "Tuning"))
    alpha_pos = np.clip(cv2.getTrackbarPos("alpha_pos(0-100)", "Tuning")/100.0, 0.01, 0.99)
    alpha_depth = np.clip(cv2.getTrackbarPos("alpha_depth(0-100)", "Tuning")/100.0, 0.01, 0.99)

    sgbm = make_sgbm(numDisp, blockSize, uniq, preFilterCap=preCap)
    disp = sgbm.compute(grayL, grayR).astype(np.float32) / 16.0
    disp[disp < 0] = 0

    max_disp = np.max(disp) if np.max(disp) > 0 else 1.0
    disp_vis8 = cv2.convertScaleAbs(disp, alpha=255.0 / max_disp)
    color_disp = cv2.applyColorMap(disp_vis8, cv2.COLORMAP_INFERNO)

    mask = np.where(disp > disp_thresh, 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = rectR.copy()
    tracked = False
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area > 800:
            x, y, w, h = cv2.boundingRect(largest)
            pad = int(0.05 * max(w, h))
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(disp.shape[1], x+w+pad), min(disp.shape[0], y+h+pad)

            roi_mask = np.zeros_like(mask_clean)
            cv2.drawContours(roi_mask, [largest], -1, 255, thickness=-1)
            roi_mask = roi_mask[y1:y2, x1:x2]

            disp_roi = disp[y1:y2, x1:x2]
            valid_idx = np.where((roi_mask == 255) & (disp_roi > 0))
            num_valid = valid_idx[0].shape[0]

            if num_valid >= MIN_VALID_POINTS:
                points_3d = cv2.reprojectImageTo3D(disp, Q)
                pts_roi = points_3d[y1:y2, x1:x2]
                selected_pts = pts_roi[valid_idx]

                finite_mask = np.isfinite(selected_pts[:,2]) & (selected_pts[:,2] > 0)
                good_pts = selected_pts[finite_mask]
                if good_pts.shape[0] >= MIN_VALID_POINTS:
                    median_X = np.median(good_pts[:,0])
                    median_Y = np.median(good_pts[:,1])
                    median_Z = np.median(good_pts[:,2])

                    M = cv2.moments(largest)
                    cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else (x1 + x2)//2
                    cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else (y1 + y2)//2

                    if smoothed_pos is None:
                        smoothed_pos = np.array([cx, cy], dtype=np.float32)
                    else:
                        smoothed_pos = alpha_pos * np.array([cx, cy], dtype=np.float32) + (1.0 - alpha_pos) * smoothed_pos

                    if smoothed_Z is None:
                        smoothed_Z = float(median_Z)
                    else:
                        smoothed_Z = alpha_depth * float(median_Z) + (1.0 - alpha_depth) * smoothed_Z

                    spx, spy = int(smoothed_pos[0]), int(smoothed_pos[1])
                    cv2.circle(output, (spx, spy), 8, (0, 255, 0), 2)
                    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 200, 0), 2)
                    depth_text = f"Z = {smoothed_Z:.3f} m"
                    cv2.putText(output, depth_text, (spx + 12, spy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    tracked = True
                else:
                    tracked = False

    if not tracked:
        if smoothed_pos is not None:
            smoothed_pos = (0.98 * smoothed_pos)
        if smoothed_Z is not None:
            smoothed_Z = smoothed_Z * 1.002

    now = time.time()
    fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
    prev_time = now

    cv2.putText(color_disp := cv2.applyColorMap(cv2.convertScaleAbs(disp, alpha=255.0/max_disp), cv2.COLORMAP_INFERNO),
                f"FPS: {fps:.1f}  disp_thresh:{disp_thresh}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Rectified Right", rectR)
    cv2.imshow("Disparity (color)", color_disp)
    cv2.imshow("Mask (clean)", mask_clean)
    cv2.imshow("Tracked (stabilized)", output)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('s'):
        ts = int(time.time())
        cv2.imwrite(os.path.join(OUT_DIR, f"tracked_{ts}.png"), output)
        print(f"Saved snapshot to {OUT_DIR}")

capL.release()
capR.release()
cv2.destroyAllWindows()
