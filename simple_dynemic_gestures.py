import os
import cv2
import numpy as np
from time import time
from keras.models import model_from_json


json_file = open('models/nn/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("models/nn/model.h5")

labels_dict = {"hands_down": 0,
               "stop": 1,
               "hands_up": 2,
               "hands_up_small": 3,
               "hands_down_small": 4,
               "hads_down_up": 5,
               "hands_to_sides": 6}


class GesturesChecker:
    dymanic_gestures_dict = {("hands_down", "hands_down_small", "hands_down"): "come_on",
                             ("stop", ""): "stop",

                             ("hads_down_up", "hands_down_small", "hads_down_up"): "go_away",
                             ("hands_to_sides", "hands_up_small", "hands_up"): "look_at_me"}

    def __init__(self, gesture_duration=5):
        self.__gestures_list = []
        self.__times_list = []

        self.__gesture_duration = gesture_duration

    @property
    def gestures_list(self):
        return self.__gestures_list

    @property
    def times_list(self):
        return self.__times_list

    def update_gestures(self, new_gesture):
        if not self.gestures_list_is_empty:
            if self.gestures_list[-1] == new_gesture:
                return
        update_time = time()
        self.__gestures_list.append(new_gesture)
        self.__times_list.append(update_time)

    def check_gestures(self):
        joined_timeline = "/".join(self.__gestures_list)
        for gest_key in self.dymanic_gestures_dict.keys():
            if gest_key == ("stop"):
                print("/".join(gest_key))
                print(joined_timeline)
            joined_gest_k = "/".join(gest_key)
            if joined_gest_k in joined_timeline:
                self.__gestures_list = []
                self.__times_list = []
                return self.dymanic_gestures_dict[gest_key]

    def check_and_remove_old_gestures(self):
        if self.gestures_list_is_empty:
            return
        current_time = time()
        actual_gestures = []
        actual_times = []
        for i in range(len(self.__gestures_list)):
            if current_time - self.__times_list[i] <= self.__gesture_duration:
                actual_gestures.append(self.__gestures_list[i])
                actual_times.append(self.__times_list[i])
        self.__times_list = actual_times
        self.__gestures_list = actual_gestures

    @property
    def gestures_list_is_empty(self):
        return not bool(len(self.gestures_list))


def predict(model, input_data):
    data = np.array([input_data])
    prediction_result = np.argmax(model.predict(data))

    labels_dict = {"hands_down": 0,
                   "stop": 1,
                   "hands_up": 2,
                   "hands_up_small": 3,
                   "hands_down_small": 4,
                   "hads_down_up": 5,
                   "hands_to_sides": 6}

    decode_labels_dict = {val: key for key, val in labels_dict.items()}
    return decode_labels_dict[prediction_result]


def rescale_values(values: np.array):
    min_val = np.min(values)
    max_val = np.max(values)
    delta = max_val - min_val

    values -= min_val
    values /= delta
    values *= 100
    values[values < 0] = 0
    return values


def rescale_pose(pose):
    points = pose.points
    xs = np.array([pt.x for pt in points])
    ys = np.array([pt.y for pt in points])
    xs = rescale_values(xs)
    ys = rescale_values(ys)
    rescaled_points = [Point(x, y) for x, y in zip(xs, ys)]
    rescaled_pose = Pose(rescaled_points)
    return rescaled_pose


def find_distance(pt1, pt2):
    return np.sqrt((pt2.x - pt1.x) ** 2 + (pt2.y - pt1.y) ** 2)


def new_extract_features(pose, central_point_idx, specific_points):
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


import cv2

from src.PoseExtractor import PoseExtractor
from src.visualization import *

from src.data_processing.pose_utils import *


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


def signed_features_extraction(pose, central_point_idx, specific_points=None):
    assert specific_points is not None, "Udefined specific points!"
    central_point = pose.points[central_point_idx]
    if central_point.x > 0 and central_point.y > 0:  # if point detected. (if (0, 0) - hidden point)
        xs = [pt.x for pt in pose.points]
        ys = [pt.y for pt in pose.points]

        dxs = [central_point.x - x for x in xs]
        dys = [central_point.y - y for y in ys]

        for i, (dx, dy) in enumerate(zip(dxs, dys)):
            if dx == central_point.x and dy == central_point.y and i in specific_points:
                return None

        dxs -= np.mean(dxs)
        dxs /= np.std(dxs)
        dys -= np.mean(dys)
        dys /= np.std(dys)

        features = []
        for point_idx in specific_points:
            features.append(dxs[point_idx])
            features.append(dys[point_idx])
        return np.array(features)
    else:
        return None


cam = cv2.VideoCapture(0)
pose_extractor = PoseExtractor()

# top_pose_points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18]
top_pose_points = [0, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18]

central_point_idx = 1
normalize_features = False
ignore_nan = True
pose_rescaling = True

gesutures_checker = GesturesChecker()

while True:
    ret, img = cam.read()
    if not ret:
        break
    img_to_show = img.copy()
    poses = pose_extractor.extract_poses_from_image(img)
    biggest_pose = get_biggest_pose(poses)
    img_to_show[:55, :250] = (255, 255, 255)
    img_to_show[-30:, -100:] = (0, 0, 0)
    cv2.putText(img_to_show, "SVM", (570, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    gesutures_checker.check_and_remove_old_gestures()

    d_g = None

    if biggest_pose is not None:

        signed_f = signed_features_extraction(biggest_pose, central_point_idx, specific_points=top_pose_points)

        features = new_extract_features(biggest_pose, 1, top_pose_points)
        if features is not None and not any(np.isnan(features)):
            gesture = predict(model, features)
            gesutures_checker.update_gestures(gesture)
            d_g = gesutures_checker.check_gestures()

            # cv2.putText(img_to_show, gesture, (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 40, 50), 2)
            cv2.putText(img_to_show, d_g, (5, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 40, 50), 2)

        draw_pose(img_to_show, biggest_pose)

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img_to_show)
    if d_g is None:
        wait_time = 10
    else:
        wait_time = 1000
    k = cv2.waitKey(wait_time)
    if k & 0xFF == 27:
        break

cv2.destroyAllWindows()
cam.release()

