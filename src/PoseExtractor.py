import os
import sys
from src.image_objects.Point import Point


openpose_path = "/home/user/Soft/openpose"
python_openpose_path = "build/python/"
full_path_to_openpose_python = os.path.join(openpose_path, python_openpose_path)
model_folder = os.path.join(openpose_path, "models")

sys.path.append(full_path_to_openpose_python)
from openpose import pyopenpose as op


class PoseExtractor:
    def __init__(self):
        self.__open_pose_wrapper = op.WrapperPython()
        self.__open_pose_wrapper.configure({"model_folder": model_folder})
        self.__open_pose_wrapper.start()
        self.__datum = op.Datum()

    def extract_poses_from_image(self, img):
        self.__update_open_pose_datum(img)
        raw_coordinates = self.__get_poses()
        poses = [self.__raw_coordinates_to_points(pose) for pose in raw_coordinates]
        return poses

    def __update_open_pose_datum(self, img):
        self.__datum.cvInputData = img
        self.__open_pose_wrapper.emplaceAndPop([self.__datum])

    def __get_poses(self):
        return self.__datum.poseKeypoints

    @staticmethod
    def __raw_coordinates_to_points(pose):
        pose_points_coordinates = [pt[:2] for pt in pose]
        k_points = [Point(*coordinate) for coordinate in pose_points_coordinates]
        return k_points

    # def draw_pose(self):
    #     return self.datum.cvOutputData
