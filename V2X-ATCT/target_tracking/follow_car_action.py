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
import utils.common_utils as cu
import core.obj_insert as oi
import turning_action as ta



def gen_data_base_on_changed_data(i=2,
        save_path = '',
        dx_val = 0.3,
        dy_val = 0.3,
        gen_all_flag=False,
        v2x_dataset_path = ''

         ): 
    
    
    now = datetime.now()
    formatted = now.strftime("%Y-%m-%d_%H:%M:%S")
    OP_TIMES = 1
    # load dataset config
    dataset_config = Config2(dataset="v2x_dataset", scene=i,dataset_path=v2x_dataset_path)
    dataset_config.v2x_dataset_saved_dir = \
        os.path.join(save_path,f'{formatted}')
        # f"{dataset_config.dataset_root}/results/test/{formatted}"
    if gen_all_flag == True:
        dataset_config.v2x_dataset_saved_dir = save_path

    if not os.path.exists(dataset_config.v2x_dataset_saved_dir):
        os.makedirs(dataset_config.v2x_dataset_saved_dir)

    
    scene_data_num = dataset_config.scene_data_num
    index_list = list(range(dataset_config.begin_index,
                            dataset_config.begin_index + scene_data_num))
    
   
    followed_car_list = ['ego','cp']
    followed_car=random.choice(followed_car_list)
    
    position_list = []
    degree_list = []

  

    for bg_index in index_list:
        ego_info = V2XInfo(bg_index, dataset_config=dataset_config)
        cp_info = V2XInfo(bg_index, is_ego=False, dataset_config=dataset_config)
        
        op_count = 0

        if bg_index%5 == 0:
            dx_val = 0.3
            dy_val = 0.3
             
            dx_val = random.uniform(-0.15, 0.15) + dx_val
            dy_val = random.uniform(-0.15, 0.15) + dy_val

        
        if bg_index == dataset_config.begin_index:
            insert_info = tcu.follow_car_action(followed_car=followed_car,init_insert=True,ego_info=ego_info,cp_info=cp_info,dx_val=dx_val,dy_val=dy_val)
        else :
            insert_info = tcu.follow_car_action(followed_car=followed_car,insert_info=insert_info,ego_info=ego_info,cp_info=cp_info,dx_val=dx_val,dy_val=dy_val)
        

        if followed_car == 'cp':
            position=insert_info['transformed_position']
            rz_degree = insert_info['transformed_rzdegree']

            flag,position = tcu.avoid_collision2(ego_info=ego_info,cp_info=cp_info,insert_position=position)
            if flag == True:
                z = oi.select_road_height(cp_info.road_pc,position)
                insert_info['insert_position']=list(cu.center_system_transform([position[0],position[1],z], ego_info.param['lidar_pose'], cp_info.param['lidar_pose']))[:2]

        else:

            position=insert_info['insert_position']

            flag,position = tcu.avoid_collision2(ego_info=ego_info,cp_info=cp_info,insert_position=position)
            if flag == True:
                insert_info['insert_position'] = position
            
            rz_degree = insert_info['rzdegree']
            # print(tar_index)

        
        position_list.append(position)
        degree_list.append(rz_degree)


        


        # transformations times of each point cloud frame
        while op_count < OP_TIMES:
            

            transformation = "insert"
            
            

            success_flag = False
           
            if transformation == "insert":
                
                success_flag, v2x_ego_id, v2x_cp_id = \
                trans.vehicle_insert_with_position_and_degree(ego_info,cp_info,position,car_degree = rz_degree,objs_index=3)
                
            
            # if not success_flag:
            #     continue

            op_count += 1

        
        ego_info.save_data_and_label(f"v2x_scene{i}_followcar",True)
        cp_info.save_data_and_label(f"v2x_scene{i}_followcar",True)

        
    #保存坐标和角度
    with open(f'{dataset_config.v2x_dataset_saved_dir}/{i}_pos_{followed_car}.json', 'w') as file:
        json.dump(position_list, file)
    with open(f'{dataset_config.v2x_dataset_saved_dir}/{i}_deg_{followed_car}.json', 'w') as file:
        json.dump(degree_list, file) 

    return dataset_config.v2x_dataset_saved_dir,formatted



def main(i,
        save_path = '',
        dx_val = 0.3,
        dy_val = 0.3,
        gen_all_flag=False,
        gen_data_for_sharding = False,
        sharding=30,
        gen_data_for_base_line = False,
        index_list_for_baseline = [],
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
    
   
    followed_car_list = ['ego','cp']
    followed_car=random.choice(followed_car_list)

    position_list = []
    degree_list = []


    if gen_data_for_sharding ==  True:
        
        index_list,start_ratio = ta.select_random_subsequence(sequence=index_list,n=sharding)
        
        print(index_list)
    
    
    if gen_data_for_base_line == True:
        index_list = index_list_for_baseline

    # for bg_index in index_list:
    for index, bg_index in enumerate(index_list):
        ego_info = V2XInfo(bg_index, dataset_config=dataset_config)
        cp_info = V2XInfo(bg_index, is_ego=False, dataset_config=dataset_config)
        
        op_count = 0

       
        if index == 0:
            insert_info = tcu.follow_car_action(followed_car=followed_car,init_insert=True,ego_info=ego_info,cp_info=cp_info,dx_val=dx_val,dy_val=dy_val)
        else :
            insert_info = tcu.follow_car_action(followed_car=followed_car,insert_info=insert_info,ego_info=ego_info,cp_info=cp_info,dx_val=dx_val,dy_val=dy_val)
        

        if followed_car == 'cp':
            position=insert_info['transformed_position']
            rz_degree = insert_info['transformed_rzdegree']

            flag,position = tcu.avoid_collision(ego_info=ego_info,cp_info=cp_info,insert_position=position)
            if flag == True:
                z = oi.select_road_height(cp_info.road_pc,position)
                insert_info['insert_position']=list(cu.center_system_transform([position[0],position[1],z], ego_info.param['lidar_pose'], cp_info.param['lidar_pose']))[:2]

        else:

            position=insert_info['insert_position']

            flag,position = tcu.avoid_collision(ego_info=ego_info,cp_info=cp_info,insert_position=position)
            if flag == True:
                insert_info['insert_position'] = position
            rz_degree = insert_info['rzdegree']
            # print(tar_index)

        

        position_list.append(position)
        degree_list.append(rz_degree)


        


        # # transformations times of each point cloud frame
        while op_count < OP_TIMES:
            

            transformation = "insert"
            
            

            success_flag = False
           
            if transformation == "insert":
                
                success_flag, v2x_ego_id, v2x_cp_id = \
                trans.vehicle_insert_with_position_and_degree(ego_info,cp_info,position,car_degree = rz_degree,objs_index=3)
                
            
            if not success_flag:
                continue

            op_count += 1

        
        ego_info.save_data_and_label(f"v2x_scene{i}_followcar",True)
        cp_info.save_data_and_label(f"v2x_scene{i}_followcar",True)

        
    with open(f'{dataset_config.v2x_dataset_saved_dir}/{i}_pos_{followed_car}.json', 'w') as file:
        json.dump(position_list, file)
    with open(f'{dataset_config.v2x_dataset_saved_dir}/{i}_deg_{followed_car}.json', 'w') as file:
        json.dump(degree_list, file) 

    return dataset_config.v2x_dataset_saved_dir,formatted,index_list


if __name__ == '__main__':

    # main(1)
    gen_data_base_on_changed_data()
   
