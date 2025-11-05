import cv2
import os
import time

CAM_LEFT_ID = 2
CAM_RIGHT_ID = 1
SAVE_PATH = "data/"

os.makedirs(os.path.join(SAVE_PATH, "left"), exist_ok=True)
os.makedirs(os.path.join(SAVE_PATH, "right"), exist_ok=True)

cap_left = cv2.VideoCapture(CAM_LEFT_ID, cv2.CAP_DSHOW)
time.sleep(1)
cap_right = cv2.VideoCapture(CAM_RIGHT_ID, cv2.CAP_DSHOW)

if not cap_left.isOpened() or not cap_right.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

print("Cameras opened successfully.")
print("Disabling auto features and applying manual settings...\n")

for cam in [cap_left, cap_right]:
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cam.set(cv2.CAP_PROP_AUTO_WB, 0)
    time.sleep(0.5)

LEFT_EXPOSURE = -6
RIGHT_EXPOSURE = -11
LEFT_WB = 5000
RIGHT_WB = 4800

cap_left.set(cv2.CAP_PROP_EXPOSURE, LEFT_EXPOSURE)
cap_right.set(cv2.CAP_PROP_EXPOSURE, RIGHT_EXPOSURE)
cap_left.set(cv2.CAP_PROP_WB_TEMPERATURE, LEFT_WB)
cap_right.set(cv2.CAP_PROP_WB_TEMPERATURE, RIGHT_WB)

time.sleep(1)

print("Applied Camera Settings:")
print(f"Left Camera:  Exposure={cap_left.get(cv2.CAP_PROP_EXPOSURE):.2f}, "
      f"WB={cap_left.get(cv2.CAP_PROP_WB_TEMPERATURE):.0f}")
print(f"Right Camera: Exposure={cap_right.get(cv2.CAP_PROP_EXPOSURE):.2f}, "
      f"WB={cap_right.get(cv2.CAP_PROP_WB_TEMPERATURE):.0f}\n")

print("="*30)
print("  Press 'SPACE' to save image pair")
print("  Press 'q' to quit")
print("="*30 + "\n")

img_counter = 0

while True:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("Error: Can't receive frame. Exiting ...")
        break

    combined = cv2.hconcat([frame_left, frame_right])
    cv2.imshow("Stereo View (Left | Right)", combined)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Quitting capture.")
        break

    elif key == ord(' '):
        img_name_left = os.path.join(SAVE_PATH, "left", f"left_{img_counter:04d}.png")
        img_name_right = os.path.join(SAVE_PATH, "right", f"right_{img_counter:04d}.png")

        cv2.imwrite(img_name_left, frame_left)
        cv2.imwrite(img_name_right, frame_right)

        print(f"Saved pair #{img_counter}:")
        print(f"    {img_name_left}")
        print(f"    {img_name_right}\n")
        img_counter += 1

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
print("Capture session closed.")
