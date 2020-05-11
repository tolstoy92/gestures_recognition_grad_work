import numpy as np

from src.image_objects.Pose import Pose
from src.image_objects.Point import Point


def get_distance_between_points(pt1, pt2):
    x1, y1 = pt1.xy
    x2, y2 = pt2.xy
    return np.sqrt(((x2 - x1)**2) + ((y2 - y1)**2))


def find_distances(pose, central_point_idx, specific_points=None):
    central_point = pose.points[central_point_idx]
    if specific_points is None:
        src_points_idxs = range(len(pose.points))
    else:
        src_points_idxs = specific_points
    if not all(central_point.int_xy):
        return np.array([np.nan for i in range(len(src_points_idxs) - 1)])
    distances = []
    for current_point_idx in src_points_idxs:
        if current_point_idx != central_point_idx:
            current_point = pose.points[current_point_idx]
            if all(current_point.int_xy):
                distance = get_distance_between_points(central_point, current_point)
            else:
                distance = np.nan
            distances.append(distance)
    return np.array(distances)


def normalize_distances(distances, ignore_nan: bool):
    get_mean = np.nanmean if ignore_nan else np.mean
    get_std = np.nanstd if ignore_nan else np.std

    normalized = distances.copy()
    mean = get_mean(normalized)
    normalized -= mean
    std = get_std(normalized)
    normalized /= std
    return normalized


def rescale_values(values: np.array):
    max_val = np.max(values)
    min_val = np.min(values)
    scale = max_val - min_val
    rescaled_values = values.copy()
    rescaled_values -= min_val
    rescaled_values /= scale
    rescaled_values *= 100
    return rescaled_values


def rescale_points(points):
    x_array = np.array([pt.x for pt in points])
    y_array = np.array([pt.y for pt in points])
    rescaled_x = rescale_values(x_array)
    rescaled_y = rescale_values(y_array)
    return [Point(x, y) for x, y in zip(rescaled_x, rescaled_y)]


def rescale_pose(pose: Pose):
    rescaled_pose_points = rescale_points(pose.points)
    return Pose(rescaled_pose_points)


def extract_features(pose, central_point_idx, normalization: bool, pose_rescaling: bool,
                     ignore_nan=None, specific_points=None):
    if normalization:
        assert isinstance(ignore_nan, bool), "Specify ignore_nan for normalization!"

    if pose_rescaling:
        pose = rescale_pose(pose)

    distances = find_distances(pose, central_point_idx, specific_points)
    if not normalization:
        features = distances
    else:
        features = normalize_distances(distances, ignore_nan)
    return features
