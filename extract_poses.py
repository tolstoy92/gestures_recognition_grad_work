import os
import cv2
import numpy as np
import pickle

from time import time

from src.image_objects.Point import Point
from src.image_objects.Pose import Pose
from src.PoseExtractor import PoseExtractor
from src.data_processing.pose_utils import *


pose_extracotr = PoseExtractor()

top_pose_points = [0, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18]
central_point_index = 1


def get_biggest_pose(poses):
    max_distance = -1
    biggest_pose = None
    if len(poses):
        for pose in poses:
            pose_key_point1 = pose.points[1]
            pose_key_point2 = pose.points[8]
            if all(pose_key_point1.int_xy) and all(pose_key_point2.int_xy):
                distance = get_distance_between_points(pose_key_point1, pose_key_point2)
                if distance >= max_distance:
                    max_distance = distance
                    biggest_pose = pose
        return biggest_pose

    
def extract_features(pose, central_point_idx, specific_points):
    assert central_point_idx not in specific_points
    
    rescaled_pose = rescale_pose(pose)
    central_point = rescaled_pose.points[central_point_idx]
    
    features = []
    
    for current_points_idx in specific_points:
        current_point = rescaled_pose.points[current_points_idx]

        dx = central_point.x - current_point.x
        dy = central_point.y - current_point.y
    
        features.append(dx)
        features.append(dy)
    
    return np.array(features)

    
def extract_features_sequence_from_video(path_to_video, counter=None):
    cap = cv2.VideoCapture(path_to_video)
    
    k_points_sequence_original = []
    k_points_sequence_flipped = []
    while True:
        ret, img = cap.read()
        if not ret:
            cap.release()
            return k_points_sequence_original, k_points_sequence_flipped
        
        poses = pose_extracotr.extract_poses_from_image(img)
        actual_pose = get_biggest_pose(poses)
        
        if actual_pose is not None:
        
            xs = np.array([p.x for p in actual_pose.points])
            ys = np.array([p.y for p in actual_pose.points])

            h, w, _ = img.shape

            x_cnt = w // 2
            y_cnt = h // 2

            d = x_cnt - xs
            flipped_x =  x_cnt + d

            flipped_points = [Point(x, y) for x, y in zip(flipped_x, ys)]

            flipped_pose = Pose(flipped_points)


            k_points_sequence_original.append([p.xy for p in actual_pose.points])
            k_points_sequence_flipped.append([p.xy for p in flipped_pose.points])


            flipped_image = cv2.flip(img, 1)
    #         poses_flipped = pose_extracotr.extract_poses_from_image(flipped_image)
    #         actual_pose_flipped = get_biggest_pose(poses_flipped)
            actual_pose_flipped = flipped_pose
            if actual_pose is None or actual_pose_flipped is None:
                continue

            for p in actual_pose.points:
                cv2.circle(img, (int(p.x), int(p.y)), 3, (255, 0, 0))

            for p in actual_pose_flipped.points:
                cv2.circle(flipped_image, (int(p.x), int(p.y)), 3, (255, 0, 0))

            if counter is not None:
                cv2.putText(img, str(counter), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100))
        else:
            k_points_sequence_original.append(None)
            k_points_sequence_flipped.append(None)
#         to_show = np.hstack([img, flipped_image])
#         cv2.imshow("img", to_show)
#         k = cv2.waitKey(1)
        
#         if k & 0xFF == 27:
#             break
            
    
            
video_dir = "/home/user/Desktop/videos/videos"


for d in os.listdir(video_dir):
    
    print("\n\n\n\n", d, "\n\n\n\n")
    if ".py" in d or ".ipynb" in d:
            continue
    gesture_folder = d

    full_features_list_from_all_come_on = []
    full_features_list_from_all_come_on_flipped = []


    full_dir_path = os.path.join(video_dir, gesture_folder) 


    video_counter = 0

    k_points_dict = {}

    
    countt = 1
    for f_name in os.listdir(full_dir_path):
        print(countt)
        
        countt += 1
        actual_video_path = os.path.join(full_dir_path, f_name)
        kp, kp_flipped = extract_features_sequence_from_video(actual_video_path, video_counter)
        k_points_dict[f_name] = kp
        k_points_dict["flipped_" + f_name] = kp_flipped

        with open('dicts/{}.pickle'.format(gesture_folder), 'wb') as handle:
            pickle.dump(k_points_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #     full_features_list_from_all_come_on.append(features)
    #     full_features_list_from_all_come_on_flipped.append(features_flipped)

        video_counter += 1
cv2.destroyAllWindows()