import cv2
import numpy as np
import time
import os

CAM_LEFT_ID, CAM_RIGHT_ID = 2, 1
CALIB_FILE = "stereo_maps.npz"
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

with np.load(CALIB_FILE) as data:
    mapL1, mapL2, mapR1, mapR2 = data['mapL1'], data['mapL2'], data['mapR1'], data['mapR2']

capL, capR = cv2.VideoCapture(CAM_LEFT_ID), cv2.VideoCapture(CAM_RIGHT_ID)
for c in (capL, capR):
    c.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    c.set(cv2.CAP_PROP_FPS, 30)
time.sleep(1.0)

cv2.namedWindow("Tuning", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tuning", 640, 240)
def nothing(x): pass
cv2.createTrackbar("numDisp(x16)", "Tuning", 6, 12, nothing)
cv2.createTrackbar("blockSize", "Tuning", 5, 15, nothing)
cv2.createTrackbar("uniqueness", "Tuning", 10, 30, nothing)
cv2.createTrackbar("K (K-Means)", "Tuning", 3, 8, nothing)
cv2.createTrackbar("disp_thresh", "Tuning", 10, 200, nothing)
cv2.createTrackbar("morph_k", "Tuning", 5, 25, nothing)

def make_sgbm(numDisp, blockSize, uniqueness):
    if blockSize % 2 == 0: blockSize += 1
    P1, P2 = 8 * 3 * blockSize**2, 32 * 3 * blockSize**2
    return cv2.StereoSGBM_create(minDisparity=0, numDisparities=numDisp, blockSize=blockSize,
                                 P1=P1, P2=P2, uniquenessRatio=uniqueness,
                                 speckleWindowSize=100, speckleRange=32,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

def segment_kmeans(image, K, morph_k):
    pixels = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, _ = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    label_img = label.reshape((FRAME_HEIGHT, FRAME_WIDTH))
    center_label = label_img[FRAME_HEIGHT // 2, FRAME_WIDTH // 2]
    mask = np.where(label_img == center_label, 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
prev_time = time.time()

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not (retL and retR): break
    rectL = cv2.remap(frameL, mapL1, mapL2, cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, mapR1, mapR2, cv2.INTER_LINEAR)
    disp_thresh = cv2.getTrackbarPos("disp_thresh", "Tuning")
    morph_k = max(1, cv2.getTrackbarPos("morph_k", "Tuning"))
    grayL, grayR = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
    numDisp = max(16, cv2.getTrackbarPos("numDisp(x16)", "Tuning") * 16)
    blockSize = cv2.getTrackbarPos("blockSize", "Tuning")
    uniq = cv2.getTrackbarPos("uniqueness", "Tuning")
    sgbm = make_sgbm(numDisp, blockSize, uniq)
    disp = sgbm.compute(grayL, grayR).astype(np.float32) / 16.0
    disp[disp < 0] = 0
    mask_depth = np.where(disp > disp_thresh, 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    mask_depth_clean = cv2.morphologyEx(cv2.morphologyEx(mask_depth, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    depth_output = cv2.bitwise_and(rectR, rectR, mask=mask_depth_clean)
    K = max(2, cv2.getTrackbarPos("K (K-Means)", "Tuning"))
    mask_kmeans_clean = segment_kmeans(rectR, K, morph_k)
    kmeans_output = cv2.bitwise_and(rectR, rectR, mask=mask_kmeans_clean)
    fgMask = backSub.apply(rectR)
    _, mask_mog2 = cv2.threshold(fgMask, 254, 255, cv2.THRESH_BINARY)
    mask_mog2_clean = cv2.morphologyEx(cv2.morphologyEx(mask_mog2, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    mog2_output = cv2.bitwise_and(rectR, rectR, mask=mask_mog2_clean)
    now = time.time(); fps = 1.0 / (now - prev_time) if now != prev_time else 0.0; prev_time = now
    cv2.putText(rectR, f"Original (FPS: {fps:.1f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(kmeans_output, f"K-Means (1-Cam, K={K})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(mog2_output, "MOG2 (1-Cam, Motion)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)
    cv2.putText(depth_output, f"Depth (2-Cam, thresh={disp_thresh})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    q_h, q_w = FRAME_HEIGHT // 2, FRAME_WIDTH // 2
    def resize_for_grid(img, h=q_h, w=q_w):
        h_img, w_img = img.shape[:2]
        scale = min(w / w_img, h / h_img)
        h_new, w_new = int(h_img * scale), int(w_img * scale)
        img_resized = cv2.resize(img, (w_new, h_new))
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        y_off, x_off = (h - h_new)//2, (w - w_new)//2
        canvas[y_off:y_off+h_new, x_off:x_off+w_new] = img_resized
        return canvas
    top = np.hstack((resize_for_grid(rectR), resize_for_grid(kmeans_output)))
    bottom = np.hstack((resize_for_grid(mog2_output), resize_for_grid(depth_output)))
    grid = np.vstack((top, bottom))
    cv2.imshow("Segmentation Comparison (Bot-L: MOG2, Bot-R: Depth)", grid)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

capL.release()
capR.release()
cv2.destroyAllWindows()
