import numpy as np


def get_distance_between_points(pt1, pt2):
    x1, y1 = pt1.xy
    x2, y2 = pt2.xy
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


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


def extract_futures(pose, central_point_idx, normalization: bool,
                    ignore_nan=None, specific_points=None):
    if normalization:
        assert isinstance(ignore_nan, bool), "Specify ignore_nan for normalization!"

    distances = find_distances(pose, central_point_idx, specific_points)
    if not normalization:
        features = distances
    else:
        features = normalize_distances(distances, ignore_nan)
    return features
