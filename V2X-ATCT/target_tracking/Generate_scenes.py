import sys
# sys.path.append('/home/maji/Downloads/V2X-ATCT/DMSTrack_master/AB3DMOT')
# sys.path.append('/home/maji/Downloads/V2X-ATCT/DMSTrack_master/AB3DMOT/Xinshuo_PyToolbox')
# sys.path.append('/home/maji/Downloads/V2X-ATCT/DMSTrack_master/V2V4Real')
# sys.path.append('/home/maji/Downloads/V2X-ATCT/DMSTrack_master')
# sys.path.append('/home/maji/Downloads/V2X-ATCT/DMSTrack_master/DMSTrack')
# sys.path.append("/home/maji/Downloads/V2X-ATCT/V2X-ATCT/target_tracking")
from pathlib import Path
import os
import argparse
import fusion_followcar_overtake_action as overtake
import follow_car_action as followcar
import turning_action as turning
# os.chdir('/home/maji/Downloads/V2X-ATCT/')

current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.parent 
print(project_root)
os.chdir(project_root)
def gen_scene(
                    scene_num=1,#1-9
                    save_path='',
                    action='Overtaking',
                    track_num=1,
                    speed=60,
                    carnum=3
                      ):

    actions = ['Overtaking','Vehicle_Following','Turning']
    
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
    print(action)
    # print(scene_action_support)

    action_id = 0
    for i,ac in enumerate(actions):
        if actions[i] == action:
            action_id = i
            break
      


    if action_id not in scene_i_support:
          print("The driving behavior is not support in this scene!")
          raise TypeError
    
    dx_val=0.3
    dy_val=0.3
    
    # return
    if speed==60:
          dx_val=0.3
          dy_val=0.3
    else :
          dx_val=0.3 + (speed-30)/200
          dx_val=0.3 + (speed-30)/200
    
    
    if action == 'Overtaking':
            # print('Overtaking')
            save_dir1,time1,index_list = overtake.main(i=scene_num,save_path=save_path,gen_data_for_sharding=False,car_num=carnum,dx_val=dx_val,dy_val=dy_val)
    elif action == 'Vehicle_Following':
        
            # print('Vehicle_Following')
            save_dir1,time1,index_list = followcar.main(i=scene_num,save_path=save_path,gen_data_for_sharding=False,car_num=carnum,dx_val=dx_val,dy_val=dy_val)
    elif action == 'Turning':
            save_dir1,time1,index_list,start_ratio = turning.main(i=scene_num,save_path=save_path,gen_data_for_sharding=False,car_num=carnum,dx_val=dx_val,dy_val=dy_val)  
    else :
        print('error')
        raise TypeError

    return True



def args_parser():
    parser = argparse.ArgumentParser(description="demo command")
    parser.add_argument('--save_path', type=str, required=False, default='/app/V2X-ATCT/target_tracking/generate_scene/'
                        )
    parser.add_argument('--scene_num', type=int, required=True,
                        )
    parser.add_argument('--driving_behaviour', type=str, required=False,default='Overtaking'
                        )
    parser.add_argument('--tracknum', type=int, required=False,
                        )
    parser.add_argument('--speed', type=int, required=False, default=60
                        )
    # parser.add_argument('--road_num', type=int, required=False,
    #                     )
    parser.add_argument('--carnum', type=int, required=False, default=3
                        )

    

    
    args = parser.parse_args()
    return args





if __name__ == '__main__':

    opt = args_parser()
    save_path = opt.save_path 
    # print(save_path)
    if save_path != None:
        save_path = os.path.dirname(save_path)

    print(save_path)

    scene_num = opt.scene_num
    action = opt.driving_behaviour
    speed = opt.speed
    carnum = opt.carnum

    gen_scene(save_path=save_path,scene_num=scene_num,action=action,speed=speed,carnum=carnum)