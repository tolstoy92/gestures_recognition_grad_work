import cv2
from time import time

from src.PoseExtractor import PoseExtractor
from src.data_processing.pose_utils import extract_futures

from src.visualization import draw_pose


cap = cv2.VideoCapture(0)

pose_extractor = PoseExtractor()

top_pose_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18]
central_point_idx = 1
normalize_features = True
ignore_nan = True


while True:
    ret, img = cap.read()
    if not ret:
        break

    start_time = time()
    poses = pose_extractor.extract_poses_from_image(img)
    for pose in poses:
        raw_features = extract_futures(pose, 1, normalization=False, specific_points=None)
        norm_features = extract_futures(pose, 1, normalization=True, ignore_nan=True, specific_points=None)
        for i, j in zip(raw_features, norm_features):
            print(i, j)
        print("\n\n")


        draw_pose(img, pose)



    end_time = time()
    fps = round(1 / (end_time - start_time), 2)

    cv2.putText(img, str(fps), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 100), 2)
    cv2.imshow("img", img)
    wait_key = cv2.waitKey(10)
    if wait_key & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
