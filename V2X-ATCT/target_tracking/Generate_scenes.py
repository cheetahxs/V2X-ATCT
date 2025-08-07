import sys
# sys.path.append('/home/maji/Downloads/V2XTargetTracking-main/DMSTrack_master/AB3DMOT')
# sys.path.append('/home/maji/Downloads/V2XTargetTracking-main/DMSTrack_master/AB3DMOT/Xinshuo_PyToolbox')
# sys.path.append('/home/maji/Downloads/V2XTargetTracking-main/DMSTrack_master/V2V4Real')
# sys.path.append('/home/maji/Downloads/V2XTargetTracking-main/DMSTrack_master')
# sys.path.append('/home/maji/Downloads/V2XTargetTracking-main/DMSTrack_master/DMSTrack')
# sys.path.append("/home/maji/Downloads/V2XTargetTracking-main/V2XGen-main/target_tracking")
import shutil
import os
import random
import argparse

from datetime import datetime
import DMSTrack.DMSTrack.main_dkf_change as dkf
import fusion_followcar_overtake_action as overtake
import follow_car_action as followcar
import turning_action as turning

import DMSTrack.V2V4Real.opencood.tools.inference_new as inference
import DMSTrack.V2V4Real.opencood.tools.inference as inference_all

import DMSTrack.AB3DMOT.main_change as cobevtmain
import DMSTrack.AB3DMOT.scripts.KITTI.evaluate as evaluate
import target_tracking.rq2.RQ2 as rq2




def gen_scene(
                    scene_num=1,#1-9
                    save_path='',
                   
                      ):

    actions = ['overtake','followcar','turning']
    
    # action = 'overtake'

    scene_action_support = [
        [0,1],
        [0,1],
        [0,1],
        [0,1,2],
        [2],
        [0,1,2],
        [1,2],
        [2],
        [2],
    ]
    scene_i_support = scene_action_support[scene_num-1]
    random.seed()
    action_i = random.choice(scene_i_support)
    action = actions[action_i]

    print(action)
    # return

    
    
    if action == 'overtake':
            print('overtake')
            save_dir1,time1,index_list = overtake.main(i=scene_num,save_path=save_path,gen_data_for_sharding=False)
    elif action == 'followcar':
        
            print('followcar')
            save_dir1,time1,index_list = followcar.main(i=scene_num,save_path=save_path,gen_data_for_sharding=False)
    elif action == 'turning':
            save_dir1,time1,index_list,start_ratio = turning.main(i=scene_num,save_path=save_path,gen_data_for_sharding=False)  
    else :
        print('error')
        raise TypeError

    return True


def args_parser():
    parser = argparse.ArgumentParser(description="demo command")
    parser.add_argument('--save_path', type=str, required=True,
                        )
    parser.add_argument('--scene_num', type=int, required=True,
                        )
    
    args = parser.parse_args()
    return args





if __name__ == '__main__':

    opt = args_parser()
    save_path = opt.save_path 
    save_path = os.path.dirname(save_path)


    scene_num = opt.scene_num

    gen_scene(save_path=save_path,scene_num=scene_num)