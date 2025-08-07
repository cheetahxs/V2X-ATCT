import math
import random
import numpy as np


def get_random_rotation(min_range=5, max_range=30):
    """
    random rotation degree, [-30,-5] or [5,30]
    """
    range1 = (-max_range, -min_range)
    range2 = (min_range, max_range)

    selected_range = range1 if np.random.rand() < 0.5 else range2

    random_rot = np.random.uniform(selected_range[0], selected_range[1])

    return random_rot


def get_insert_location(v2x_info):
    """
    Randomly generate insert coordinates based on the road split extent.
    """
    road_pc = v2x_info.road_pc
    road_x = road_pc[:, 0]
    road_y = road_pc[:, 1]
    x_range = (np.min(road_x), np.max(road_x))
    y_range = (np.min(road_y), np.max(road_y))
    pos_x = 0
    pos_y = 0

    is_valid = False
    while not is_valid:
        pos_x = random.uniform(x_range[0], x_range[1])
        pos_y = random.uniform(y_range[0], y_range[1])
        print(pos_x, pos_y)
        if math.sqrt(pos_x**2 + pos_y**2) <= 100:
            is_valid = True
    degree = get_random_rotation()

    return [pos_x, pos_y], degree
