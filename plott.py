import numpy as np
import cv2
from src.data_processing.pose_utils import *
from matplotlib import pyplot as plt
from src.PoseExtractor import PoseExtractor
from src.visualization import *

img = cv2.imread("/home/user/Desktop/wq.jpg") # shape: (960, 1280, 3)

pose_extr = PoseExtractor()

poses = pose_extr.extract_poses_from_image(img)
#
# plt.figure(figsize=(10, 7))
#
# pose1 = poses[0]
# pose2 = poses[1]
#
# rescaled_pose1 = rescale_pose(pose1)
# rescaled_pose2 = rescale_pose(pose2)
# pts1 = rescaled_pose1.points
# pts2 = rescaled_pose2.points
# #
# x1, y1 = np.array([p.x for p in pts1 if p.x]), np.array([p.y for p in pts1 if p.y])
# x2, y2 = np.array([p.x for p in pts2 if p.x]), np.array([p.y for p in pts2 if p.y])
#
# dx1 = x1[1] - 50
# dy1 = y1[1] - 50
#
# dx2 = x2[1] - 50
# dy2 = y2[1] - 50
#
# x1, x2 = x1 - dx1, x2 - dx2
# y1, y2 = y1 - dy1, y2 - dy2
#
# y1, y2 = np.clip(y1, 0, 100), np.clip(y2, 0, 100)
# # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.scatter(x1, y1, marker="^", label="pose 1", s=80)
# plt.scatter(x2, y2, marker="o", label="pose 2", s=80)
# # plt.xlim(0, 1280)
# # plt.ylim(0, 960)
# plt.legend()
# plt.xticks(range(0, 101, 10))
# plt.yticks(range(0, 101, 10))
# plt.grid()
# plt.gca().invert_yaxis()
# plt.show()
#
#

cp_img = img.copy()


# for pose in poses:
#     draw_pose(cp_img, pose)

draw_pose(cp_img, poses[0])
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.imshow("img", cp_img)
cv2.waitKey()
cv2.destroyAllWindows()

