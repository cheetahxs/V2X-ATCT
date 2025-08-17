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
import os
import argparse

from typing import List, Optional

from datetime import datetime
# import DMSTrack.main_dkf_change2 as dkf
import DMSTrack.DMSTrack.main_dkf_change as dkf
import fusion_followcar_overtake_action as overtake
import follow_car_action as followcar
import turning_action as turning

import DMSTrack.V2V4Real.opencood.tools.inference_new as inference

import yaml
import DMSTrack.V2V4Real.opencood.tools.inference as inference_all
import json
import secrets
import DMSTrack.AB3DMOT.main_change as cobevtmain
import DMSTrack.AB3DMOT.scripts.KITTI.evaluate as evaluate
import target_tracking.rq1.RQ1 as rq1

# import os
# os.chdir('/home/maji/Downloads/V2XTargetTracking-main/DMSTrack_master/DMSTrack')
# os.chdir('/home/maji/Downloads/V2XTargetTracking-main/DMSTrack_master/AB3DMOT')  


# def cal_cobevt_result(
#         scene = 1,
#         data_dir = '',
#         config_dir = './V2X-ATCT/target_tracking/intermediate_fusion',
#         fusion_method = 'intermediate',
#         feature = 'cobevt',
#         gen_npy_flag = True,
# ):
    
#     npy_path = os.path.join(data_dir,'npy')
#     if gen_npy_flag == True:
        
#         if os.path.exists(npy_path):
#             shutil.rmtree(npy_path)

#         trans_data_to_npy(data_dir=data_dir,config_dir=config_dir,fusion_method=fusion_method)
    
#     source_dir = os.path.join(npy_path,'ab3dmot_detection')
    
#     destination_dir = f'./DMSTrack/AB3DMOT/data/v2v4real/detection/cobevt_Car_val'
    
    
#     src_txt_path = os.path.join(source_dir,f'000{scene}.txt')
#     des_txt_path = os.path.join(destination_dir,f'000{scene}.txt')
#     copy_txt_file(source_file=src_txt_path,target_file=des_txt_path)


#     # copy_folder(src=source_txts, dst=destination_txts)
    
#     args = cobevtmain.parse_args()
#     cobevtmain.main(args)
    
    
#     sys.argv.append('cobevt_Car_val_H1')
#     sys.argv.append(1)
#     sys.argv.append('3D')
#     sys.argv.append(0.25)
#     sys.argv.append('0000')
#     sys.argv.append('val')

    
#     sys.argv[5]=f'000{scene-1}'
#     print(sys.argv[5])
#     evaluate.evaluate_main()

#     source_file = os.path.join('./DMSTrack/AB3DMOT/results/v2v4real/cobevt_Car_val_H1','summary_car_average_eval3D.txt')
#     dest_file = os.path.join(data_dir,f'summary_car_average_eval3D_{feature}_000{scene-1}.txt')

#     copy_txt_file(source_file, dest_file)




def read_per_eva_result(
        dir_path='',
        time = '2025-05-19_20:03:47',
        feature='',
        file_name = 'summary_car_average_eval3D.txt'
):  
    
    dir_path = os.path.join(dir_path,file_name)

    split_lines = []
    eval_result = {}

    with open(dir_path, 'r', encoding='utf-8') as file:
        for line in file:
            split_lines.append(line.strip())

            if 'Multiple' in line and 'Object' in line and 'Tracking' in line and 'Accuracy' in line and '(MOTA)' in line :
                eval_result['MOTA'] = line.strip().split()[-1]
            if 'Mostly' in line and 'Tracked' in line  :
                eval_result['MT'] = line.strip().split()[-1] 
            if 'Mostly' in line and 'Lost' in line  :
                eval_result['ML'] = line.strip().split()[-1] 
            if 'Partly' in line and 'Tracked' in line  :
                eval_result['PT'] = line.strip().split()[-1] 


            if 'False' in line and 'Positives' in line  :
                eval_result['FP'] = line.strip().split()[-1] 
            if 'False' in line and 'Negatives' in line and 'Ignored' not in line :
                eval_result['FN'] = line.strip().split()[-1] 
                print(eval_result['FN'])
            if 'ID-switches' in line  :
                eval_result['ID-switches'] = line.strip().split()[-1] 


    eval_result['sAMOTA'] = split_lines[-2].strip().split()[0]
    eval_result['AMOTA'] = split_lines[-2].strip().split()[1]
    eval_result['AMOTP'] = split_lines[-2].strip().split()[2]

    # print(split_lines)
    eval_result['time'] = time
    eval_result['feature'] = feature
    return eval_result





def gen_data(
    save_path='/V2X-ATCT/target_tracking/rq2/result',
    scene = 1,
    sharding = 30,
    gen_seed_num = 5,
    carnum=3,
    speed=60
):

    now = datetime.now()
    formatted = now.strftime("%Y-%m-%d_%H:%M:%S")

    

    save_path_guide = os.path.join(save_path,f'guide/{formatted}_seeds')
    save_path_baseline = os.path.join(save_path,'baseline')
    
    save_dir1,time1,save_dir2,time2 = gen_single_data(scene_num=scene,sharding=sharding,save_path_baseline=save_path_baseline,save_path_guide=save_path_guide,gen_seed_num=gen_seed_num,carnum=carnum,speed=speed)


    return save_path_guide,save_dir2


def gen_all_data(
    save_path_root='/V2X-ATCT/target_tracking/rq2/result',
    gen_time = 9,
    sharding = 31,
    gen_seed_num = 5,
    speed=60,
    carnum=3
):

    now = datetime.now()
    formatted = now.strftime("%Y-%m-%d_%H:%M:%S")
    save_path = os.path.join(save_path_root,f'{formatted}')


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    guide_list = []
    baseline_list = []
    scene_list = []

    for index in range(1,gen_time+1):

        try:
            save_dir1,save_dir2 = gen_data(scene=index,save_path=save_path,sharding=sharding,gen_seed_num = gen_seed_num)
        except:
            continue
        
        guide_list.append(save_dir1)
        baseline_list.append(save_dir2)
        scene_list.append(index)

    return guide_list,baseline_list,formatted,scene_list








def gen_single_data(
                    scene_num=1,#1-9
                    save_path_guide='',
                    save_path_baseline = '',
                    sharding = 30,
                    gen_seed_num = 3,
                    carnum=3,
                    speed=60
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

    if speed==60:
          dx_val=0.3,dy_val=0.3
    else :
          dx_val=0.3 + (speed-30)/200
          dx_val=0.3 + (speed-30)/200
    
    if action == 'overtake':

            print('overtake')
            save_dir1,time1,index_list = overtake.main(i=scene_num,save_path=save_path_guide,gen_data_for_sharding=False,sharding=sharding,speed=speed,car_num=carnum)
            save_dir2,time2,index_list = overtake.main(i=scene_num,save_path=save_path_baseline,gen_data_for_base_line=False,sharding=sharding,index_list_for_baseline=index_list,speed=speed,car_num=carnum)
            for index in range(gen_seed_num-1):
                overtake.main(i=scene_num,save_path=save_path_guide,gen_data_for_base_line=False,sharding=sharding,index_list_for_baseline=index_list,speed=speed,car_num=carnum)


    elif action == 'followcar':
        
            print('followcar')
            save_dir1,time1,index_list = followcar.main(i=scene_num,save_path=save_path_guide,gen_data_for_sharding=False,sharding=sharding,speed=speed,car_num=carnum)
            save_dir2,time2,index_list = followcar.main(i=scene_num,save_path=save_path_baseline,gen_data_for_base_line=False,sharding=sharding,index_list_for_baseline=index_list,speed=speed,car_num=carnum)
            for index in range(gen_seed_num-1):
                followcar.main(i=scene_num,save_path=save_path_guide,gen_data_for_base_line=False,sharding=sharding,index_list_for_baseline=index_list,speed=speed,car_num=carnum)


    elif action == 'turning':
        
            save_dir1,time1,index_list,start_ratio = turning.main(i=scene_num,save_path=save_path_guide,gen_data_for_sharding=False,sharding=sharding,speed=speed,car_num=carnum)
            # print(index_list)
            save_dir2,time2,index_list,start_ratio = turning.main(i=scene_num,save_path=save_path_baseline,gen_data_for_base_line=False,sharding=sharding,index_list_for_baseline=index_list,start_ratio=start_ratio,gen_data_for_sharding=False,speed=speed,car_num=carnum)
            for index in range(gen_seed_num-1):
                turning.main(i=scene_num,save_path=save_path_guide,gen_data_for_base_line=False,sharding=sharding,index_list_for_baseline=index_list,start_ratio=start_ratio,gen_data_for_sharding=False,speed=speed,car_num=carnum)
    else :
        print('error')
        raise TypeError

    return save_dir1,time1,save_dir2,time2
    



def trans_data_to_npy(scene= 1,data_dir='',config_dir='./V2X-ATCT/target_tracking/late_fusion',fusion_method='no_fusion_keep_all'):

    inference.main(scene=scene,config_path=config_dir,data_dir=data_dir,fusion_methon=fusion_method)



def copy_txt_file(source_file, target_file):
    try:
        with open(source_file, 'r', encoding='utf-8') as source:
            content = source.read()
        
        with open(target_file, 'w', encoding='utf-8') as target:
            target.write(content)
        
        print(f"file from {source_file} to {target_file}")
    except FileNotFoundError:
        print(f"error: can not find the files {source_file}")
    except Exception as e:
        print(f"error: {e}")

def MOT_operation(save_dir,feature='fusion',seq_eval_mode='all',sharding=30):

    dkf.DATA_DIR = save_dir
    # dkf.SEQ_EVAL_MODE = seq_eval_mode
    dkf.FEATURE = feature
    # dkf.Sharding = sharding
    dkf.SEQ_EVAL_MODE = seq_eval_mode
    args = dkf.parse_args()
    save_dirs = dkf.main(args)

    print(save_dirs)

    source_file = os.path.join(save_dirs[0],'summary_car_average_eval3D.txt')
    dest_file = os.path.join(save_dir,f'summary_car_average_eval3D_{feature}_{seq_eval_mode}.txt')

    copy_txt_file(source_file, dest_file)


    return save_dirs




def gen_data_and_test(
        
        config_dir = './V2X-ATCT/target_tracking/late_fusion',
        fusion_method = 'no_fusion_keep_all',
        only_test = False,
        # data_dir = '',
        seed_num = 5,
        save_path_root='./V2X-ATCT/target_tracking/rq2/result',
        scene_num = 9,
        sharding = 31,
        select_seed_num=1,
        insert_time=1,
        speed=60,
        carnum=3
        

):  
    if only_test == False:
        guide_list,baseline_list,time,scene_list = gen_all_data(save_path_root=save_path_root,gen_time=scene_num,sharding=sharding,gen_seed_num=seed_num,carnum=carnum,speed=speed)

    print('guide_list',guide_list)
    print('baseline_list',baseline_list)


    for index,guide_path in enumerate(guide_list):

        scene = scene_list[index]

        # dirs = []
        for item in os.listdir(guide_path):
            item_path = os.path.join(guide_path, item)
            if os.path.isdir(item_path) :
            # if os.path.isdir(item_path) and 'time' in item :
                # dirs.append(item)
                try:
                    trans_data_to_npy(scene=scene,data_dir=item_path,config_dir=config_dir,fusion_method=fusion_method)
                    MOT_operation(save_dir=item_path,feature='fusion',seq_eval_mode=f'000{scene-1}')
                    MOT_operation(save_dir=item_path,feature='bev',seq_eval_mode=f'000{scene-1}')
                    MOT_operation(save_dir=item_path,feature='pos',seq_eval_mode=f'000{scene-1}')
                    # trans_data_to_npy(scene=scene,data_dir=item_path,config_dir='./V2X-ATCT/target_tracking/intermediate_fusion',fusion_method='intermediate')
                    # rq1.cal_cobevt_result(scene=scene,data_dir=item_path)
                except:
                    print('error')
                

    for index, baseline_path in enumerate(baseline_list) :
        scene = scene_list[index]
        try:
            trans_data_to_npy(scene=scene,data_dir=baseline_path,config_dir=config_dir,fusion_method=fusion_method)
            MOT_operation(save_dir=baseline_path,feature='fusion',seq_eval_mode=f'000{scene-1}')
            MOT_operation(save_dir=baseline_path,feature='bev',seq_eval_mode=f'000{scene-1}')
            MOT_operation(save_dir=baseline_path,feature='pos',seq_eval_mode=f'000{scene-1}')
            # trans_data_to_npy(scene=scene,data_dir=baseline_path,config_dir='./V2X-ATCT/target_tracking/intermediate_fusion',fusion_method='intermediate')
            # rq1.cal_cobevt_result(scene=scene,data_dir=baseline_path)
        except:
            print('error')
    
    select_seed_and_save_result(baseline_dir=os.path.join(save_path_root,time,'baseline'),guide_dir=os.path.join(save_path_root,time,'guide'))

    print('Done!')



def read_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = yaml.safe_load(f)
            return data
        except yaml.YAMLError as e:
            print(f"YAML error: {e}")
            return None


def copy_scene_folders(src, dst, overwrite=False):
    try:
        if not os.path.exists(dst):
            os.makedirs(dst)
        
        for item in os.listdir(src):
            item_path = os.path.join(src, item)

            if os.path.isdir(item_path) and 'scene' in item:
                dest_path = os.path.join(dst, item)
                if os.path.exists(dest_path) and overwrite:
                    shutil.rmtree(dest_path)
                
                shutil.copytree(item_path, dest_path)
                print(f"success: {item_path} -> {dest_path}")
        
    except Exception as e:
        print(f"error: {e}")



def select_seed_and_save_result(baseline_dir='',guide_dir=''):


    select_seed = []

    entries = sorted(os.listdir(guide_dir))

    for entry in entries:
        entry_path = os.path.join(guide_dir, entry)
        if os.path.isdir(entry_path):
    
            items = sorted(os.listdir(entry_path))

            # seeds = {}
            min_fitness = 10
            min_fitness_path = ''
            for item in items:
                item_path = os.path.join(entry_path, item)
                if os.path.isdir(item_path):

                    data = read_yaml(os.path.join(item_path, 'eval.yaml'))
                    # ap50 = data['ap_50']
                    ai50 = data['ave_iou_50']
                    # fitness = 0.5*ap50 + 0.5*ai50
                    fitness = ai50
                    if fitness < min_fitness :
                        min_fitness = fitness
                        min_fitness_path = item_path

            select_seed.append(min_fitness_path)

    guide_cobevt_dir = os.path.join(guide_dir,'cobevt')
    for seed in select_seed:
        copy_scene_folders(src=seed,dst=guide_cobevt_dir)

    # rq1.trans_data_to_npy(data_dir=guide_cobevt_dir,config_dir='V2X-ATCT/target_tracking/intermediate_fusion',fusion_method='intermediate')
    rq1.cal_cobevt_result(data_dir=guide_cobevt_dir)

    # print(f'select seed: {select_seed}')

    guide_result_fusion = []
    guide_result_pos = []
    guide_result_bev = []
    guide_result_cobevt = []
    for item_path in select_seed:
        # print(item_path)
        fusion = read_per_eva_result(dir_path=item_path,feature='fusion',time=os.path.basename(item_path.rstrip("/")),file_name=search_files_by_name(item_path,'summary_car_average_eval3D_fusion')[0])
        pos = read_per_eva_result(dir_path=item_path,feature='pos',time=os.path.basename(item_path.rstrip("/")),file_name=search_files_by_name(item_path,'summary_car_average_eval3D_pos')[0]  )
        bev = read_per_eva_result(dir_path=item_path,feature='bev',time=os.path.basename(item_path.rstrip("/")),file_name=search_files_by_name(item_path,'summary_car_average_eval3D_bev')[0])  
        cobevt = read_per_eva_result(dir_path=guide_cobevt_dir,feature='cobevt',time=os.path.basename(item_path.rstrip("/")),file_name=search_files_by_name(item_path,'summary_car_average_eval3D_cobevt')[0])         

        guide_result_fusion.append(fusion)
        guide_result_pos.append(pos)
        guide_result_bev.append(bev)
        guide_result_cobevt.append(cobevt)

    # print(guide_result_fusion)

 
    with open(os.path.join(guide_dir,'guide_fusion.json'), "w", encoding="utf-8") as f:
        json.dump(guide_result_fusion, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(guide_dir,'guide_pos.json'), "w", encoding="utf-8") as f:
        json.dump(guide_result_pos, f, ensure_ascii=False, indent=2)

    with open(os.path.join(guide_dir,'guide_bev.json'), "w", encoding="utf-8") as f:
        json.dump(guide_result_bev, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(guide_dir,'guide_cobevt.json'), "w", encoding="utf-8") as f:
        json.dump(guide_result_cobevt, f, ensure_ascii=False, indent=2)
    
    entries2 = sorted(os.listdir(baseline_dir))
    baseline_result_fusion = []
    baseline_result_pos = []
    baseline_result_bev = []
    baseline_result_cobevt = []

    baseline_cobevt_dir = os.path.join(baseline_dir,'cobevt')
    for entry in entries2:
        entry_path = os.path.join(baseline_dir, entry)
        copy_scene_folders(src=entry_path,dst=baseline_cobevt_dir)

    rq1.cal_cobevt_result(data_dir=baseline_cobevt_dir)


    for entry in entries2:
        entry_path = os.path.join(baseline_dir, entry)
        if os.path.isdir(entry_path):
            fusion = read_per_eva_result(dir_path=entry_path,feature='fusion',time=os.path.basename(item_path.rstrip("/")),file_name=search_files_by_name(item_path,'summary_car_average_eval3D_fusion')[0])
            pos = read_per_eva_result(dir_path=entry_path,feature='pos',time=os.path.basename(item_path.rstrip("/")),file_name=search_files_by_name(item_path,'summary_car_average_eval3D_pos')[0])
            bev = read_per_eva_result(dir_path=entry_path,feature='bev',time=os.path.basename(item_path.rstrip("/")),file_name=search_files_by_name(item_path,'summary_car_average_eval3D_bev')[0])   
            cobevt = read_per_eva_result(dir_path=baseline_cobevt_dir,feature='cobevt',time=os.path.basename(item_path.rstrip("/")),file_name=search_files_by_name(item_path,'summary_car_average_eval3D_cobevt')[0])        
            baseline_result_fusion.append(fusion)
            baseline_result_pos.append(pos)
            baseline_result_bev.append(bev)
            baseline_result_cobevt.append(cobevt)

    # print(baseline_result_fusion)


    with open(os.path.join(baseline_dir,'baseline_fusion.json'), "w", encoding="utf-8") as f:
        json.dump(baseline_result_fusion, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(baseline_dir,'baseline_pos.json'), "w", encoding="utf-8") as f:
        json.dump(baseline_result_pos, f, ensure_ascii=False, indent=2)

    with open(os.path.join(baseline_dir,'baseline_bev.json'), "w", encoding="utf-8") as f:
        json.dump(baseline_result_bev, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(baseline_dir,'baseline_cobevt.json'), "w", encoding="utf-8") as f:
        json.dump(baseline_result_cobevt, f, ensure_ascii=False, indent=2)

    aa_base_fu,ap_base_fu = cal_ave_amota_amotp(baseline_result_fusion)
    aa_gui_fu,ap_gui_fu = cal_ave_amota_amotp(guide_result_fusion)
    print(f'system: fusion;')
    print(f'baseline:')
    print(f'AMOTA: {aa_base_fu};AMOTP: {ap_base_fu};')
    print('guidance:')
    print(f'AMOTA: {aa_gui_fu};AMOTP: {ap_gui_fu};')
    print('----------------------------')

    aa_base_pos,ap_base_pos = cal_ave_amota_amotp(baseline_result_pos)
    aa_gui_pos,ap_gui_pos = cal_ave_amota_amotp(guide_result_pos)
    print(f'system: pos;')
    print(f'baseline:')
    print(f'AMOTA: {aa_base_pos};AMOTP: {ap_base_pos};')
    print('guidance:')
    print(f'AMOTA: {aa_gui_pos};AMOTP: {ap_gui_pos};')
    print('----------------------------')

    aa_base_bev,ap_base_bev = cal_ave_amota_amotp(baseline_result_bev)
    aa_gui_bev,ap_gui_bev = cal_ave_amota_amotp(guide_result_bev)
    print(f'system: bev;')
    print(f'baseline:')
    print(f'AMOTA: {aa_base_bev};AMOTP: {ap_base_bev};')
    print('guidance:')
    print(f'AMOTA: {aa_gui_bev};AMOTP: {ap_gui_bev};')
    print('----------------------------')

    aa_base_co,ap_base_co = cal_ave_amota_amotp(baseline_result_cobevt)
    aa_gui_co,ap_gui_co = cal_ave_amota_amotp(guide_result_cobevt)
    print(f'system: cobevt;')
    print(f'baseline:')
    print(f'AMOTA: {aa_base_co};AMOTP: {ap_base_co};')
    print('guidance:')
    print(f'AMOTA: {aa_gui_co};AMOTP: {ap_gui_co};')
    print('----------------------------')



def cal_ave_amota_amotp(values):
    amota=float(0)
    amotp=float(0)
    count = 0
    for val in values:
        count += 1
        amota += val['AMOTA']
        amotp += val['AMOTP']


    return amota/count,amotp/count


def search_files_by_name(directory: str, keyword: str, recursive: bool = False, 
                         return_full_path: bool = False) -> List[str]:
    matched_files = []
    try:
        for root, _, files in os.walk(directory):
            for file in files:
                if keyword.lower() in file.lower():
                    file_path = os.path.join(root, file) if return_full_path else file
                    matched_files.append(file_path)
            if not recursive:
                break  
    except Exception as e:
        print(f"error: {e}")
    return matched_files



def args_parser():
    parser = argparse.ArgumentParser(description="rq2 command")
    parser.add_argument('--save_path', type=str, required=True,
                        )
    parser.add_argument('--seed_num', type=int, required=True,
                        )
    parser.add_argument('--system', type=str, required=False,
                        )
    parser.add_argument('--select_seed_num', type=int, required=False,
                        )
    parser.add_argument('--driving_behaviour', type=str, required=False,
                        )
    parser.add_argument('--insert_time', type=int, required=False,
                        )
    parser.add_argument('--speed', type=int, required=False,
                        )
    parser.add_argument('--carnum', type=int, required=False,
                        )
    # parser.add_argument('--spawnsnum', type=int, required=False,
    #                     )
    
    args = parser.parse_args()
    return args





if __name__ == '__main__':

    opt = args_parser()
    save_path = opt.save_path 
    save_path = os.path.dirname(save_path)


    seed_num = opt.seed_num
    carnum = opt.carnum
    speed = opt.speed

    gen_data_and_test(save_path_root=save_path,seed_num=seed_num,carnum=carnum,speed=speed)

   