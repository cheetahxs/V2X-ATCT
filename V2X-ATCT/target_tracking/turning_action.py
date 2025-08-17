import os , sys
# sys.path.append("/home/maji/Downloads/V2XTargetTracking-main/V2XGen-main/")
import copy
import json
import os.path
import random
import argparse
import shutil
import rq1.obj_transformation as trans

from logger import CLogger
from config.config import Config
from config.config2 import Config2
from utils.v2x_object import V2XInfo
import threading
import utils.visual as vs

from scipy.optimize import fsolve
import numpy as np
import z3_solver as zs
import pickle

import rq1.obj_transformation_ori as trans_ori
import target_tracking.tracking_common_utils as tcu
from datetime import datetime
import utils.v2X_file as vf
import core.obj_insert as oi




def gen_data_base_on_changed_data(
        #  road_index=0,
        i=5,
        save_path='',
        
        dx_val = 0.8,
        dy_val = 0.8,
        # road_index = 0,
        gen_all_flag=False,
        v2x_dataset_path = '',
        insert_time = 2,
        carnum=3
         ): 
    

    now = datetime.now()
    formatted = now.strftime("%Y-%m-%d_%H:%M:%S")
   
    # OP_TIMES = int(method)
    OP_TIMES = 1
    # load dataset config
    dataset_config = Config2(dataset="v2x_dataset", scene=i,dataset_path=v2x_dataset_path)
    dataset_config.v2x_dataset_saved_dir = \
        os.path.join(save_path,f'{formatted}')
        
    if gen_all_flag == True:
        dataset_config.v2x_dataset_saved_dir = save_path
    if not os.path.exists(dataset_config.v2x_dataset_saved_dir):
        os.makedirs(dataset_config.v2x_dataset_saved_dir)
    

    scene_data_num = dataset_config.scene_data_num
    index_list = list(range(dataset_config.begin_index,
                            dataset_config.begin_index + scene_data_num))
    
    res = tcu.read_txt(f'V2X-ATCT/target_tracking/road_information/scene{i}/path.txt')
    road_index = random.randint(0,len(res)-1)
    points = tcu.get_points(f'V2X-ATCT/target_tracking/road_information/scene{i}/',res[road_index])
    


    lane_num = 1 if i == 5 or i == 8  else 2

    points,distance = tcu.gen_init_tar_pos(points=points,lane_num=lane_num)

    if i == 4:
        start_point_index = 3 
    else :
        start_point_index = 0

    if insert_time == 2:
        start_point_index = start_point_index+1
    else:
        start_point_index = start_point_index+2

    
    position_list = []
    degree_list = []

    for bg_index in index_list:
        ego_info = V2XInfo(bg_index, dataset_config=dataset_config)
        cp_info = V2XInfo(bg_index, is_ego=False, dataset_config=dataset_config)
        
        op_count = 0

        if bg_index%5 == 0:
            dx_val = 0.8
            dy_val = 0.8
             
            dx_val = random.uniform(-0.4, 0.4) + dx_val
            dy_val = random.uniform(-0.4, 0.4) + dy_val

        
        if bg_index == dataset_config.begin_index:
            insert_info = tcu.truning_action(points=points,init_insert=True,ego_info=ego_info,cp_info=cp_info,start_point_index=start_point_index,dx_val=dx_val,dy_val=dy_val)
        else :
            insert_info = tcu.truning_action(points=points,ego_info=ego_info,cp_info=cp_info,insert_info=insert_info,dx_val=dx_val,dy_val=dy_val)

        position=insert_info['insert_position']

        flag,position = tcu.avoid_collision2(ego_info=ego_info,cp_info=cp_info,insert_position=position)

        if flag == True:

            z = oi.select_road_height(ego_info.road_pc,position)

            insert_info['insert_position'] = position
            insert_info['ins_pos_world'] = zs.ego_xyz_to_world_coordinate_require_z(position[0],position[1],z,ego_info.param['lidar_pose'])

     
        degree = insert_info['rz_degree']
        position_list.append(position)
        degree_list.append(degree)


        # # transformations times of each point cloud frame
        while op_count < OP_TIMES:
            

            transformation = "insert"
            
            

            success_flag = False
           
            if transformation == "insert":
                
                success_flag, v2x_ego_id, v2x_cp_id = \
                trans.vehicle_insert_with_position_and_degree(ego_info,cp_info,position,car_degree = degree,objs_index=carnum)
                
            
            # if not success_flag:
            #     continue

            op_count += 1

        
        ego_info.save_data_and_label(f"v2x_scene{i}_turning",True)
        cp_info.save_data_and_label(f"v2x_scene{i}_turning",True)

    with open(f'{dataset_config.v2x_dataset_saved_dir}/{i}_pos_{road_index}_{start_point_index}_{distance}.json', 'w') as file:
        json.dump(position_list, file)
    with open(f'{dataset_config.v2x_dataset_saved_dir}/{i}_deg_{road_index}_{start_point_index}_{distance}.json', 'w') as file:
        json.dump(degree_list, file) 

    return dataset_config.v2x_dataset_saved_dir,formatted











def select_random_subsequence(sequence, n):

    if n < 1 or n > len(sequence):
        return []
    
    max_start = len(sequence) - n
    start_index = random.randint(0, max_start)

    return sequence[start_index : start_index + n], int(start_index/len(sequence))





def main(
        #  road_index=0,
        i=5,
        save_path='',
        
        dx_val = 0.8,
        dy_val = 0.8,
        # road_index = 0,
        gen_all_flag=False,
        gen_data_for_sharding = False,
        sharding=30,
        gen_data_for_base_line = False,
        index_list_for_baseline = [],
        start_ratio = 0,
        car_num=3
         ): 
    
    now = datetime.now()
    formatted = now.strftime("%Y-%m-%d_%H:%M:%S")
   
    OP_TIMES = 1
    # load dataset config
    dataset_config = Config(dataset="v2x_dataset", scene=i)
    dataset_config.v2x_dataset_saved_dir = \
        os.path.join(save_path,f'{formatted}')

    if gen_all_flag == True:
        dataset_config.v2x_dataset_saved_dir = save_path
    if not os.path.exists(dataset_config.v2x_dataset_saved_dir):
        os.makedirs(dataset_config.v2x_dataset_saved_dir)
    
 
    
    scene_data_num = dataset_config.scene_data_num
    index_list = list(range(dataset_config.begin_index,
                            dataset_config.begin_index + scene_data_num))
    

    

    res = tcu.read_txt(f'V2X-ATCT/target_tracking/road_information/scene{i}/path.txt')
    road_index = random.randint(0,len(res)-1)
    points = tcu.get_points(f'V2X-ATCT/target_tracking/road_information/scene{i}/',res[road_index])


    lane_num = 1 if i == 5 or i == 8  else 2

    points,distance = tcu.gen_init_tar_pos(points=points,lane_num=lane_num)

    if i == 4:
        start_point_index = 3 
    else :
        start_point_index = 0

    position_list = []
    degree_list = []

    start_ratio = 0
    if gen_data_for_sharding ==  True:
        
        index_list,start_ratio = select_random_subsequence(sequence=index_list,n=sharding)
     

    #用于rq2随机生成
    if gen_data_for_base_line == True:
        index_list = index_list_for_baseline

        start_ratio = start_ratio

    for index, bg_index in enumerate(index_list):
        ego_info = V2XInfo(bg_index, dataset_config=dataset_config)
        cp_info = V2XInfo(bg_index, is_ego=False, dataset_config=dataset_config)
        
        op_count = 0

        
        if index == 0:
            if gen_data_for_sharding == True:
                start_point_index = -1
                finsh_flag = False
                ego_pos = zs.ego_xyz_to_world_coordinate_require_z(0,0,0,ego_info.param['lidar_pose'])
                cp_pos = zs.ego_xyz_to_world_coordinate_require_z(0,0,0,cp_info.param['lidar_pose'])
                for index, point in enumerate(points) :
                    ego_to_init_ins_dis = tcu.cal_x1y1_2_x2y2_distance(ego_pos[0],ego_pos[1],point[0],point[1])
                    cp_to_init_ins_dis = tcu.cal_x1y1_2_x2y2_distance(cp_pos[0],cp_pos[1],point[0],point[1])
                    if ego_to_init_ins_dis < 30 or cp_to_init_ins_dis < 30 :
                        start_point_index = index
                        finsh_flag = True
                        break
                if finsh_flag == False:
                    start_point_index = int(start_ratio * len(points))

            insert_info = tcu.truning_action(points=points,init_insert=True,ego_info=ego_info,cp_info=cp_info,start_point_index=start_point_index,dx_val=dx_val,dy_val=dy_val)

        else :
            insert_info = tcu.truning_action(points=points,ego_info=ego_info,cp_info=cp_info,insert_info=insert_info,dx_val=dx_val,dy_val=dy_val)
            

        position=insert_info['insert_position']

        flag,position = tcu.avoid_collision(ego_info=ego_info,cp_info=cp_info,insert_position=position)

        if flag == True:

            z = oi.select_road_height(ego_info.road_pc,position)

            insert_info['insert_position'] = position
            insert_info['ins_pos_world'] = zs.ego_xyz_to_world_coordinate_require_z(position[0],position[1],z,ego_info.param['lidar_pose'])

     
        degree = insert_info['rz_degree']
        
        
        position_list.append(position)
        degree_list.append(degree)

        
        while op_count < OP_TIMES:
            

            transformation = "insert"
            
            

            success_flag = False
           
            if transformation == "insert":
                
                success_flag, v2x_ego_id, v2x_cp_id = \
                trans.vehicle_insert_with_position_and_degree(ego_info,cp_info,position,car_degree = degree,objs_index=car_num)
                
            
            # if not success_flag:
            #     continue

            op_count += 1

        
        ego_info.save_data_and_label(f"v2x_scene{i}_turning",True)
        cp_info.save_data_and_label(f"v2x_scene{i}_turning",True)

    with open(f'{dataset_config.v2x_dataset_saved_dir}/{i}_pos_{road_index}_{start_point_index}_{distance}.json', 'w') as file:
        json.dump(position_list, file)
    with open(f'{dataset_config.v2x_dataset_saved_dir}/{i}_deg_{road_index}_{start_point_index}_{distance}.json', 'w') as file:
        json.dump(degree_list, file) 

    return dataset_config.v2x_dataset_saved_dir,formatted,index_list,start_ratio








if __name__ == '__main__':


    main(i=4,save_path='')