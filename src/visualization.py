import cv2


def draw_point(img, point, size=5, color=(255, 0, 100)):
    cv2.circle(img, point.int_xy, size, color, -1)


def draw_line(img, line_point1, lint_point_2, color=(0, 255, 255), thickness=2):
    cv2.line(img, line_point1.int_xy, lint_point_2.int_xy, color, thickness)


POSE_LINES_COLORS = {(0, 1): (140, 255, 0),
                     (1, 8): (0, 0, 255),
                     (8, 9): (150, 255, 0),
                     (8, 12): (255, 0, 190),
                     (9, 10): (90, 255, 120),
                     (12, 13): (255, 120, 120),
                     (10, 11): (130, 255, 180),
                     (13, 14): (170, 120, 120),
                     (11, 24): (180, 255, 180),
                     (14, 21): (170, 0, 120),
                     (11, 22): (180, 255, 280),
                     (14, 19): (170, 0, 120),
                     (22, 23): (180, 255, 180),
                     (19, 20): (170, 0, 120),
                     (1, 2): (0, 255, 255),
                     (1, 5): (255, 255, 0),
                     (2, 3): (0, 155, 255),
                     (5, 6): (255, 150, 0),
                     (3, 4): (0, 155, 180),
                     (6, 7): (255, 150, 200),
                     (0, 15): (255, 255, 255),
                     (0, 16): (255, 255, 255),
                     (15, 17): (255, 255, 255),
                     (16, 18): (255, 255, 255)}


def draw_pose(img, pose):
    right_side = True  # right and left side are red and green (green and red)
    for points_idxs, line_color in POSE_LINES_COLORS.items():
        right_side = not right_side
        idx1, idx2 = points_idxs
        pt1, pt2 = pose.points[idx1], pose.points[idx2]
        if all(pt1.int_xy) and all(pt2.int_xy):
            draw_line(img, pt1, pt2, color=line_color)
            point_color = (0, 255, 0) if right_side else (0, 0, 255)
            draw_point(img, pt1, color=point_color)
            draw_point(img, pt2, color=point_color)
