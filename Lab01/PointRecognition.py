import numpy as np

def is_point_on_positive_side(normal, point):
    return np.dot(point, normal) > 0

def is_point_on_propper_side(normal, point, value):
    return is_point_on_positive_side(normal, point) == (value > 0)

def all_points_on_propper_side(normal, points: np.ndarray, values: np.ndarray):
    print(points, points.shape[0], values.shape[0])
    if points.shape[0] != values.shape[0]:
        raise ValueError("Size mismatch: 'points' and 'values' must have the same number of elements.")

    for point, value in zip(points, values):
        if not is_point_on_propper_side(normal, point, value):
            return False
    return True