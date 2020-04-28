import cv2
from time import time

from src.PoseExtractor import PoseExtractor


cap = cv2.VideoCapture(0)

pose_extractor = PoseExtractor()

while True:
    ret, img = cap.read()
    if not ret:
        break

    start_time = time()
    poses = pose_extractor.extract_poses_from_image(img)
    h, w, _ = img.shape
    for pose in poses:
        for pt in pose:
            cv2.circle(img, pt.int_xy, 5, (100, 0, 255), -1)

    end_time = time()
    fps = round(1 / (end_time - start_time), 2)

    cv2.putText(img, str(fps), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 100), 2)
    cv2.imshow("img", img)
    wait_key = cv2.waitKey(10)
    if wait_key & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
