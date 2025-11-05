import cv2
import numpy as np
import open3d as o3d
import time

CAM_LEFT_ID, CAM_RIGHT_ID = 2, 1
CALIB_FILE = "stereo_maps.npz"
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
DISP_SCALE = 16.0
MIN_POINTS = 400

try:
    with np.load(CALIB_FILE) as f:
        mapL1, mapL2 = f["mapL1"], f["mapL2"]
        mapR1, mapR2 = f["mapR1"], f["mapR2"]
        Q = f["Q"]
    print("Loaded calibration file.")
except Exception as e:
    raise SystemExit(f"Could not load calibration: {e}")

capL = cv2.VideoCapture(CAM_LEFT_ID, cv2.CAP_DSHOW)
capR = cv2.VideoCapture(CAM_RIGHT_ID, cv2.CAP_DSHOW)
for c in (capL, capR):
    c.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    c.set(cv2.CAP_PROP_FPS, 30)
if not (capL.isOpened() and capR.isOpened()):
    raise SystemExit("Could not open both cameras.")
time.sleep(1.0)

cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Controls", 500, 300)
def nothing(x): pass
cv2.createTrackbar("numDisp(x16)", "Controls", 6, 12, nothing)
cv2.createTrackbar("blockSize", "Controls", 7, 15, nothing)
cv2.createTrackbar("uniqueness", "Controls", 10, 30, nothing)
cv2.createTrackbar("disp_thresh", "Controls", 3, 20, nothing)
cv2.createTrackbar("voxel_size", "Controls", 6, 30, nothing)
cv2.createTrackbar("min_depth_cm", "Controls", 3, 30, nothing)
cv2.createTrackbar("max_depth_cm", "Controls", 140, 300, nothing)

use_wls = False
try:
    import cv2.ximgproc as ximgproc
    use_wls = True
    print("Using cv2.ximgproc WLS filtering for disparity.")
except Exception:
    print("cv2.ximgproc not available — using plain SGBM disparity.")

def make_sgbm(num_disp, block_size, uniq, preFilterCap=63):
    if block_size % 2 == 0: block_size += 1
    P1 = 8 * 3 * block_size ** 2
    P2 = 32 * 3 * block_size ** 2
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1, P2=P2,
        disp12MaxDiff=1,
        preFilterCap=preFilterCap,
        uniquenessRatio=uniq,
        speckleWindowSize=80,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Dense 3D Voxels", width=960, height=720)
voxel_geom = None
show_voxel_grid = True

prev_time = time.time()

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not (retL and retR):
        print("Frame grab failed, retrying...")
        time.sleep(0.02)
        continue
    rectL = cv2.remap(frameL, mapL1, mapL2, cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, mapR1, mapR2, cv2.INTER_LINEAR)
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    numDisp = max(16, cv2.getTrackbarPos("numDisp(x16)", "Controls") * 16)
    blockSize = max(3, cv2.getTrackbarPos("blockSize", "Controls"))
    uniqueness = cv2.getTrackbarPos("uniqueness", "Controls")
    disp_thresh = max(1, cv2.getTrackbarPos("disp_thresh", "Controls"))
    voxel_size_mm = max(4, cv2.getTrackbarPos("voxel_size", "Controls"))
    voxel_size = voxel_size_mm / 1000.0
    min_depth = cv2.getTrackbarPos("min_depth_cm", "Controls") / 100.0
    max_depth = cv2.getTrackbarPos("max_depth_cm", "Controls") / 100.0

    sgbm = make_sgbm(numDisp, blockSize, uniqueness)

    if use_wls:
        right_matcher = ximgproc.createRightMatcher(sgbm)
        wls_filter = ximgproc.createDisparityWLSFilter(sgbm)
        wls_filter.setLambda(8000.0)
        wls_filter.setSigmaColor(1.5)
        dispL = sgbm.compute(grayL, grayR)
        dispR = right_matcher.compute(grayR, grayL)
        dispL = np.int16(dispL)
        dispR = np.int16(dispR)
        try:
            filtered_disp = wls_filter.filter(dispL, grayL, None, dispR)
        except Exception:
            filtered_disp = dispL
        disp = filtered_disp.astype(np.float32) / DISP_SCALE
    else:
        dispL = sgbm.compute(grayL, grayR)
        disp = dispL.astype(np.float32) / DISP_SCALE

    disp[np.isnan(disp)] = 0
    disp[disp < 0] = 0

    points_3d = cv2.reprojectImageTo3D(disp, Q)
    z = points_3d[:, :, 2]
    mask_valid = (disp > disp_thresh) & np.isfinite(z) & (z > min_depth) & (z < max_depth)
    pts = points_3d[mask_valid]
    colors = cv2.cvtColor(rectR, cv2.COLOR_BGR2RGB)[mask_valid]

    print("Points:", pts.shape[0], "Voxel size (m):", voxel_size)

    if pts.shape[0] > MIN_POINTS:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pts)
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        try:
            cloud, ind = cloud.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
        except Exception:
            pass
        cloud_down = cloud.voxel_down_sample(voxel_size=voxel_size)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(cloud_down, voxel_size=voxel_size)
        vis.clear_geometries()
        vis.add_geometry(voxel_grid)
        vis.reset_view_point(True)
        voxel_geom = voxel_grid

    vis.poll_events()
    vis.update_renderer()

    disp8 = cv2.convertScaleAbs(disp, alpha=255.0 / (np.max(disp) + 1e-6))
    color_disp = cv2.applyColorMap(disp8, cv2.COLORMAP_INFERNO)
    combo = np.hstack((rectR, color_disp))
    now = time.time()
    fps = 1.0 / (now - prev_time + 1e-9)
    prev_time = now
    cv2.putText(combo, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.imshow("Rectified + Disparity", combo)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting.")
        break

capL.release()
capR.release()
try:
    vis.destroy_window()
except Exception:
    pass
cv2.destroyAllWindows()
print("Clean exit.")
