import os
import numpy as np
import open3d as o3d
from utils.common_utils import pc_numpy_2_o3d


def load_road_split_labels(label_path):
    labels = np.fromfile(label_path, dtype=np.uint32).reshape((-1, 1))
    return labels


def split_pc(labels):
    # print(len(labels))
    inx_road_arr = []   
    inx_other_road_arr = [] 
    inx_other_ground_arr = []   
    inx_no_road_arr = []    
    '''
        40: "road"          
        44: "parking"       
        48: "sidewalk"      
        49: "other-ground"  
        72: "terrain"       
    '''
    for i in range(len(labels)):
        # print(i)
        lb = labels[i][0]
        if lb == 40:
            inx_road_arr.append(i)  
        elif lb == 44:
            inx_other_road_arr.append(i)
        elif lb == 48:
            inx_other_road_arr.append(i)    
        elif lb in (70, 71):
            inx_other_ground_arr.append(i) 
        else:
            inx_no_road_arr.append(i)   
    # print(inx_road_arr)
    return inx_road_arr, inx_other_road_arr, inx_other_ground_arr, inx_no_road_arr


def road_split(pc, road_pc_path, road_label_path):
    pc_path = road_pc_path
    label_path = road_label_path

    if os.path.exists(pc_path):
        labels = load_road_split_labels(label_path)
        # print(labels[0])
        road_pc = np.fromfile(pc_path, dtype=np.float32).reshape((-1, 3))
        # print(road_pc)
        inx_road_arr, inx_other_road_arr, inx_other_ground_arr, inx_no_road_arr = split_pc(labels)
        # print(inx_other_road_arr)
        _pc_non_road = pc[inx_other_road_arr + inx_other_ground_arr + inx_no_road_arr]
        # print(_pc_non_road)
    else:
        labels = load_road_split_labels(label_path)

        inx_road_arr, inx_other_road_arr, inx_other_ground_arr, inx_no_road_arr = split_pc(labels)
        if len(inx_road_arr) <= 10:
            return None, None, None, None

        _pc_road, _pc_other_road, _pc_other_ground, _pc_no_road = \
            pc[inx_road_arr], pc[inx_other_road_arr], pc[inx_other_ground_arr], pc[inx_no_road_arr]

        _pc_non_road = pc[inx_other_road_arr + inx_other_ground_arr + inx_no_road_arr]
                
        pcd_road = pc_numpy_2_o3d(_pc_road)

        cl, ind = pcd_road.remove_radius_outlier(nb_points=7, radius=1)
        pcd_inlier_road = pcd_road.select_by_index(ind)

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_inlier_road, 10)

        pcd_inter = mesh.sample_points_uniformly(number_of_points=50000)

        _pc_inter = np.asarray(pcd_inter.points)
        dis = np.linalg.norm(_pc_inter, axis=1, ord=2)
        _pc_inter_valid = _pc_inter[dis > 4]
        
        road_pc = _pc_inter_valid.astype(np.float32)
        road_pc.astype(np.float32).tofile(pc_path, )

    return road_pc, _pc_non_road, labels


if __name__ == '__main__':
    ...
