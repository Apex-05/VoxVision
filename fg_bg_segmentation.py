import cv2
import numpy as np
import time
import os

CAM_LEFT_ID = 2
CAM_RIGHT_ID = 1
CALIB_FILE = "stereo_maps.npz"
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

OUT_DIR = "seg_output"
os.makedirs(OUT_DIR, exist_ok=True)

try:
    with np.load(CALIB_FILE) as data:
        mapL1, mapL2 = data['mapL1'], data['mapL2']
        mapR1, mapR2 = data['mapR1'], data['mapR2']
    print(f"Loaded calibration from {CALIB_FILE}")
except Exception as e:
    print(f"Could not load '{CALIB_FILE}': {e}")
    raise SystemExit

capL = cv2.VideoCapture(CAM_LEFT_ID)
capR = cv2.VideoCapture(CAM_RIGHT_ID)
if not (capL.isOpened() and capR.isOpened()):
    print("Could not open both cameras. Check CAM IDs.")
    raise SystemExit

for cap in (capL, capR):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

time.sleep(1.0)

cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Controls", 420, 300)

def nothing(x): pass

cv2.createTrackbar("numDisp(x16)", "Controls", 6, 12, nothing)
cv2.createTrackbar("blockSize", "Controls", 5, 15, nothing)
cv2.createTrackbar("uniqueness", "Controls", 10, 30, nothing)
cv2.createTrackbar("nearThresh", "Controls", 10, 200, nothing)
cv2.createTrackbar("farThresh", "Controls", 200, 500, nothing)
cv2.createTrackbar("morphOpen", "Controls", 3, 20, nothing)
cv2.createTrackbar("morphClose", "Controls", 7, 40, nothing)
cv2.createTrackbar("alpha(0-100)", "Controls", 70, 100, nothing)

print("\nControls:")
print(" - Press 'g' to toggle GrabCut refinement.")
print(" - Press 's' to save a screenshot of the current outputs.")
print(" - Press 'q' to quit.\n")

def make_sgbm(numDisp, blockSize, uniqueness, preFilterCap=31):
    if blockSize % 2 == 0: blockSize += 1
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

use_guided = False
try:
    import cv2.ximgproc as ximgproc
    use_guided = True
    print("Using ximgproc.guidedFilter for edge-preserving smoothing.")
except Exception:
    print("ximgproc not available — falling back to bilateral smoothing on disparity.")

grabcut_refine = False
running_mask = None
frame_idx = 0
prev_time = time.time()

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        print("Frame capture failed.")
        break

    rectL = cv2.remap(frameL, mapL1, mapL2, cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, mapR1, mapR2, cv2.INTER_LINEAR)

    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    numDisp = max(16, cv2.getTrackbarPos("numDisp(x16)", "Controls") * 16)
    blockSize = cv2.getTrackbarPos("blockSize", "Controls")
    blockSize = blockSize + 1 if blockSize % 2 == 0 else blockSize
    uniqueness = cv2.getTrackbarPos("uniqueness", "Controls")
    nearThresh = cv2.getTrackbarPos("nearThresh", "Controls")
    farThresh = cv2.getTrackbarPos("farThresh", "Controls")
    morphOpen = cv2.getTrackbarPos("morphOpen", "Controls")
    morphClose = cv2.getTrackbarPos("morphClose", "Controls")
    alpha_pct = cv2.getTrackbarPos("alpha(0-100)", "Controls")
    alpha = np.clip(alpha_pct / 100.0, 0.01, 0.99)

    sgbm = make_sgbm(numDisp=numDisp, blockSize=blockSize, uniqueness=uniqueness)
    disp = sgbm.compute(grayL, grayR).astype(np.float32) / 16.0

    disp_vis = disp.copy()
    disp_vis[disp_vis < 0] = 0
    max_disp_vis = np.max(disp_vis) if np.max(disp_vis) > 0 else 1.0
    disp_8u = cv2.convertScaleAbs(disp_vis, alpha=(255.0 / max_disp_vis))
    color_disp = cv2.applyColorMap(disp_8u, cv2.COLORMAP_INFERNO)

    disp_filtered = disp.copy()
    if use_guided:
        guide = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        disp_u8 = cv2.convertScaleAbs(np.clip(disp, 0, max_disp_vis), alpha=(255.0 / max_disp_vis))
        disp_filtered_u8 = ximgproc.guidedFilter(guide, disp_u8, radius=7, eps=1e-2)
        disp_filtered = disp_filtered_u8.astype(np.float32) * (max_disp_vis / 255.0)
    else:
        disp_filtered = cv2.bilateralFilter(disp_8u, d=7, sigmaColor=75, sigmaSpace=75).astype(np.float32) * (max_disp_vis / 255.0)

    near = max(0, min(int(nearThresh), int(np.max(disp_filtered))))
    far = max(0, min(int(farThresh), int(np.max(disp_filtered))))
    if far < near:
        near, far = far, near

    mask = np.logical_and(disp_filtered >= near, disp_filtered <= far).astype(np.uint8) * 255

    if morphOpen > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphOpen, morphOpen))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    if morphClose > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphClose, morphClose))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    num_labels, labels = cv2.connectedComponents(mask)
    if num_labels > 1:
        max_label = 1
        max_size = 0
        for lbl in range(1, num_labels):
            n = np.sum(labels == lbl)
            if n > max_size:
                max_size = n
                max_label = lbl
        mask_largest = (labels == max_label).astype(np.uint8) * 255
        mask = mask_largest

    mask_refined = mask.copy()
    if grabcut_refine:
        gc_mask = np.where(mask == 255, cv2.GC_PR_FGD, cv2.GC_BGD).astype('uint8')
        ys, xs = np.where(mask == 255)
        if ys.size > 0 and xs.size > 0:
            y0, y1 = max(0, ys.min()-10), min(mask.shape[0]-1, ys.max()+10)
            x0, x1 = max(0, xs.min()-10), min(mask.shape[1]-1, xs.max()+10)
            rect = (x0, y0, x1-x0, y1-y0)
        else:
            rect = (10, 10, mask.shape[1]-20, mask.shape[0]-20)
            gc_mask[:] = cv2.GC_BGD
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        try:
            cv2.grabCut(rectL, gc_mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
            mask_refined = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        except Exception as e:
            cv2.grabCut(rectL, gc_mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
            mask_refined = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')

    mask_f = (mask_refined.astype(np.float32) / 255.0)
    if running_mask is None:
        running_mask = mask_f.copy()
    else:
        running_mask = alpha * mask_f + (1.0 - alpha) * running_mask

    mask_smooth = (running_mask > 0.5).astype(np.uint8) * 255

    mask_3c = cv2.merge([mask_smooth, mask_smooth, mask_smooth])
    foreground = cv2.bitwise_and(rectL, mask_3c)
    bg_blur = cv2.GaussianBlur(rectL, (31,31), 0)
    inv_mask = cv2.bitwise_not(mask_smooth)
    inv_mask_3c = cv2.merge([inv_mask, inv_mask, inv_mask])
    background_part = cv2.bitwise_and(bg_blur, inv_mask_3c)
    composite = cv2.add(foreground, background_part)

    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0
    prev_time = curr_time

    cv2.putText(color_disp, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(color_disp, f"near={near} far={far} alpha={alpha:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    cv2.imshow("Rectified Right (RGB)", rectR)
    cv2.imshow("Rectified Left (RGB)", rectL)
    cv2.imshow("Disparity (color)", color_disp)
    cv2.imshow("Raw Mask (threshold)", mask)
    cv2.imshow("Smoothed Mask", mask_smooth)
    cv2.imshow("Foreground Composite", composite)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('g'):
        grabcut_refine = not grabcut_refine
        print(f"GrabCut refinement toggled to: {grabcut_refine}")
    elif key == ord('s'):
        ts = int(time.time())
        cv2.imwrite(os.path.join(OUT_DIR, f"composite_{ts}.png"), composite)
        cv2.imwrite(os.path.join(OUT_DIR, f"mask_{ts}.png"), mask_smooth)
        print(f"Saved snapshots to '{OUT_DIR}'")

    frame_idx += 1

capL.release()
capR.release()
cv2.destroyAllWindows()
