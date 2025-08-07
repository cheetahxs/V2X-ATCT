import pickle
import os , sys
sys.path.append("/home/maji/Downloads/V2XTargetTracking-main/V2XGen-main/")
import yaml
import math
import numpy as np
from scipy.interpolate import interp1d
import utils.v2X_file as vf
import numpy as np

import open3d as o3d
import target_tracking.z3_solver as zs
import utils.common_utils as cu
import core.obj_insert as oi
import target_tracking.tracking_common_utils as tcu


road_edge = 19#场景4主干道宽度

#随机选取主干道中心线两点
#计算两点连线方程
#利用约束求解器计算道路边界点
#约束：距离中心线[a,b]，在中心线的上、下方，
def gen_park_position(scene=4,
                      road_dir='same',#same or different道路方向是否相同
                      ):

    points = gen_main_road_points(scene=scene)

    # index = np.random.randint(0,len(points)-2)
    index = 9

    
    p1 = points[index]
    p2 = points[index+1]
    k1= (p1[1]-p2[1])/(p1[0]-p2[0])
    b1= p1[1]-k1*p1[0]

    a = road_edge/2 -2#道路边界范围
    b = road_edge/2 -1
    
    x,y = zs.find_point(k1,b1,a,b,road_dir,[p1[1],p2[1]] if p1[1]<p2[1] else [p2[1],p1[1]] )

    x = float(x)
    y = float(y)

    x_1 = x+0.5
    b_1 = y - k1*x
    y_1 = x_1*k1 + b_1#用于标记终点方向用

   



    return x,y,x_1,y_1


def gen_main_road_points(scene=4):

    road_labels = tcu.read_txt(f'target_tracking/scene_pcd_world_coordinate/scene{scene}/path.txt')
    # points = tcu.get_points('target_tracking/scene_pcd_world_coordinate/scene4/',res[7])
    # print(res)

    main_road_list = ''

    for label in road_labels:
        if 'main_road_list' in label:
            main_road_list  = label

    # print(main_road_list)
    points = tcu.get_points(f'target_tracking/scene_pcd_world_coordinate/scene{scene}/',main_road_list)

    # print(points)

    return points


if __name__ == '__main__':

    # gen_road_edge_points()
    # gen_main_road_points()
    x,y = gen_park_position()
    print(x,y)