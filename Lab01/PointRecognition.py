import numpy as np

# TODO: ONLY WORKS WITH BIAS = 0!!!! Function would have to change in that case
def is_point_on_positive_side(normal, point):
    normal_without_bias = normal[1:]
    return np.dot(point, normal_without_bias) > 0

def is_point_on_propper_side(normal, point, value):
    return is_point_on_positive_side(normal, point) == (value > 0)

def all_points_on_propper_side(normal, points: np.ndarray, values: np.ndarray):
    if points.shape[0] != values.shape[0]:
        raise ValueError("Size mismatch: 'points' and 'values' must have the same number of elements.")

    for point, value in zip(points, values):
        if not is_point_on_propper_side(normal, point, value):
            return False
    return True