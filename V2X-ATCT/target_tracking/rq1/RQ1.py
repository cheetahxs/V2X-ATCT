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



# os.chdir('/home/maji/Downloads/V2XTargetTracking-main/DMSTrack_master/DMSTrack')
# os.chdir('/home/maji/Downloads/V2XTargetTracking-main/DMSTrack_master/AB3DMOT')  
# os.chdir('/home/maji/Downloads/V2XTargetTracking-main/DMSTrack_master/AB3DMOT')



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




def print_and_save(eval_results,
                   save_path,
                   feature
                   ):
    min_MOTA = 1
    min_MT = 1
    max_ML = 0
    min_sAMOTA = 1
    min_AMOTA = 1
    min_AMOTP = 1

    min_MOTA_num = 0
    min_MT_num = 0
    max_ML_num = 0
    min_sAMOTA_num = 0
    min_AMOTA_num = 0
    min_AMOTP_num = 0
    
    for index, per_eval_result in enumerate(eval_results):
        if min_MOTA > float(per_eval_result['MOTA']):
            min_MOTA = float(per_eval_result['MOTA'])
            min_MOTA_num = index
        if min_MT > float(per_eval_result['MT']):
            min_MT = float(per_eval_result['MT'])
            min_MT_num = index
        if max_ML < float(per_eval_result['ML']):
            max_ML = float(per_eval_result['ML'])
            max_ML_num = index
        if min_sAMOTA > float(per_eval_result['sAMOTA']):
            min_sAMOTA = float(per_eval_result['sAMOTA'])
            min_sAMOTA_num = index
        if min_AMOTA > float(per_eval_result['AMOTA']):
            min_AMOTA = float(per_eval_result['AMOTA'])
            min_AMOTA_num = index
        if min_AMOTP > float(per_eval_result['AMOTP']):
            min_AMOTP = float(per_eval_result['AMOTP'])
            min_AMOTP_num = index

    for index, per_eval_result in enumerate(eval_results):
        if index == min_sAMOTA_num:
            per_eval_result['sAMOTA'] = '('+per_eval_result['sAMOTA']+')'
        else :
            per_eval_result['sAMOTA'] = ' '+per_eval_result['sAMOTA']+' '
        if index == min_AMOTA_num:
            per_eval_result['AMOTA'] = '('+per_eval_result['AMOTA']+')'
        else :
            per_eval_result['AMOTA'] = ' '+per_eval_result['AMOTA']+' '
        if index == min_AMOTP_num:
            per_eval_result['AMOTP'] = '('+per_eval_result['AMOTP']+')'
        else:
            per_eval_result['AMOTP'] = ' '+per_eval_result['AMOTP']+' '
        if index == min_MOTA_num:
            per_eval_result['MOTA'] = '('+per_eval_result['MOTA']+')'
        else:
            per_eval_result['MOTA'] = ' '+per_eval_result['MOTA']+' '
        if index == min_MT_num:
            per_eval_result['MT'] = '('+per_eval_result['MT']+')'
        else:
            per_eval_result['MT'] = ' '+per_eval_result['MT']+' '
        if index == max_ML_num:
            per_eval_result['ML'] = '('+per_eval_result['ML']+')'
        else:
            per_eval_result['ML'] = ' '+per_eval_result['ML']+' '
        
        per_eval_result['PT'] = ' '+per_eval_result['PT']+' '
        per_eval_result['FP'] = ' '+per_eval_result['FP']+' '
        per_eval_result['FN'] = ' '+per_eval_result['FN']+' '
        per_eval_result['ID-switches'] = ' '+per_eval_result['ID-switches']+' '

    
   

    with open(os.path.join(save_path,f'result_{feature}.txt'), "w", encoding="utf-8") as file:
        for index,per_eval_result in enumerate(eval_results):
            # if index%3 == 0:
            if index == 0:
                file.write('====================================================================================================================' + "\n")  # 每行末尾加换行符
                file.write('time                              feature sAMOTA  AMOTA   AMOTP   MOTA     MT       ML      PT   FP   FN ID-switches' + "\n") 
            file.write(per_eval_result['time']+ ' ' + \
                       per_eval_result['feature'].rjust(6)+ \
              per_eval_result['sAMOTA']+\
              per_eval_result['AMOTA']+\
              per_eval_result['AMOTP']+\
              per_eval_result['MOTA']+\
              per_eval_result['MT']+\
              per_eval_result['ML'] + \
              per_eval_result['PT'] + \
              per_eval_result['FP'] + \
              per_eval_result['FN'] + \
              per_eval_result['ID-switches'] + \
                "\n") 

    return  True






def gen_all_data(
    SAVE_PATH_ROOT='./V2X-ATCT/target_tracking/rq1/result/',
    time=1,
    v2x_dataset_path = '',
    carnum=3,
    speed=60,
):
 
    now = datetime.now()
    formatted = now.strftime("%Y-%m-%d_%H:%M:%S")
    SAVE_PATH = os.path.join(SAVE_PATH_ROOT,f'{formatted}_{time}_time')

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    for i in range(1,10):
        gen_single_data(scene_num=i,SAVE_PATH=SAVE_PATH,time = time,v2x_dataset_path=v2x_dataset_path,carnum=carnum,speed=speed)


    return SAVE_PATH



def gen_data_and_test(
        config_dir = './V2X-ATCT/target_tracking/late_fusion',
        fusion_method = 'no_fusion_keep_all',
        only_test = False,
        data_dir = '',
        time = 1

):  
    if only_test == False:
        data_dir = gen_all_data(time=time,SAVE_PATH_ROOT='./V2X-ATCT/target_tracking/rq1/result/')

    trans_data_to_npy(data_dir=data_dir,config_dir=config_dir,fusion_method=fusion_method)
    MOT_operation(save_dir=data_dir)




def gen_single_data(
                    scene_num=1,#1-9
                    SAVE_PATH='',
                    speed=60,
                    carnum=3,
                    time = 2,
                    v2x_dataset_path = '/home/maji/Downloads/V2XTargetTracking-main/V2XGen-main/target_tracking/rq1/init_seeds/2/v2x_dataset'
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
    action_i = random.choice(scene_i_support)
    action = actions[action_i]

    SCENE_NUM = scene_num

    if speed==60:
          dx_val=0.3,dy_val=0.3
    else :
          dx_val=0.3 + (speed-30)/200
          dx_val=0.3 + (speed-30)/200
    
    if action == 'overtake':
        save_dir,time = overtake.gen_data_base_on_changed_data(i=SCENE_NUM,save_path=SAVE_PATH,gen_all_flag=True,v2x_dataset_path=v2x_dataset_path,carnum=carnum,dx_val=dx_val,dy_val=dy_val)
        
    elif action == 'followcar':          
        save_dir,time = followcar.gen_data_base_on_changed_data(i=SCENE_NUM,save_path=SAVE_PATH,gen_all_flag=True,v2x_dataset_path=v2x_dataset_path,carnum=carnum,dx_val=dx_val,dy_val=dy_val)
        

    elif action == 'turning':
        save_dir,time = turning.gen_data_base_on_changed_data(i=SCENE_NUM,save_path=SAVE_PATH,gen_all_flag=True,insert_time=time,v2x_dataset_path=v2x_dataset_path,carnum=carnum,dx_val=dx_val,dy_val=dy_val)
        
    return True 






def trans_data_to_npy(data_dir='',config_dir='',fusion_method=''):

    inference_all.main(data_dir=data_dir,config_path=config_dir,fusion_method=fusion_method)

    return True 


def copy_txt_file(source_file, target_file):
    try:
        with open(source_file, 'r', encoding='utf-8') as source:
            content = source.read()
        with open(target_file, 'w', encoding='utf-8') as target:
            target.write(content)
        
        print(f"files from {source_file} to {target_file}")
    except FileNotFoundError:
        print(f"error: can not find the files {source_file}")
    except Exception as e:
        print(f"error: {e}")

def MOT_operation(save_dir,time='',feature='fusion',seq_eval_mode='all'):
    dkf.DATA_DIR = save_dir
    dkf.SEQ_EVAL_MODE = seq_eval_mode
    dkf.FEATURE = feature
    dkf.TIME = time

    args = dkf.parse_args()
    save_dirs = dkf.main(args)

    source_file = os.path.join(save_dirs[0],'summary_car_average_eval3D.txt')
    dest_file = os.path.join(save_dir,f'summary_car_average_eval3D_{feature}_{seq_eval_mode}.txt')

    copy_txt_file(source_file, dest_file)
    return save_dirs



 
def get_result_in_single_model(
        data_dir = '/home/maji/Downloads/V2XTargetTracking-main/V2XGen-main/target_tracking/rq1/results/all',
        time='',
        feature = 'fusion',
        # file_name = 'summary_car_average_eval3D.txt',
                   ):
    
    results = []
    for i in range(9):


        file_name = f'summary_car_average_eval3D_{feature}_000{i}.txt'
        result = read_per_eva_result(dir_path=data_dir,time=time,feature=feature,file_name=file_name)
        results.append(result)
    
    print_and_save(eval_results=results,save_path=data_dir,feature=feature)
    

   
def get_all_result(
        data_dir = '/home/maji/Downloads/V2XTargetTracking-main/V2XGen-main/target_tracking/rq1/results/all',
        feature = 'fusion',
        # file_name = 'summary_car_average_eval3D.txt',
                   ):

    file_name = f'summary_car_average_eval3D_{feature}.txt'

    dirs = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) :
        # if os.path.isdir(item_path) and 'time' in item :
            dirs.append(item)

    results = []
    for dir in dirs:
        
        dir_path = os.path.join(data_dir,dir)

        result = read_per_eva_result(dir_path=dir_path,time=dir,feature=feature,file_name=file_name)

        results.append(result)
    
    print_and_save(eval_results=results,save_path=data_dir,feature=feature)

def copy_folder(src, dst, overwrite=False):
    try:
        if os.path.exists(dst) and overwrite:
            shutil.rmtree(dst)
            
        shutil.copytree(src, dst)
        print(f"success: {src} -> {dst}")
        
    except Exception as e:
        print(f"error: {e}")


def cal_cobevt_result(
        data_dir = '',
        config_dir = './V2X-ATCT/target_tracking/intermediate_fusion',
        fusion_method = 'intermediate',
        feature = 'cobevt',
        gen_npy_flag = True,
):
    
    npy_path = os.path.join(data_dir,'npy')
    if gen_npy_flag == True:
        
        if os.path.exists(npy_path):
            shutil.rmtree(npy_path)

        trans_data_to_npy(data_dir=data_dir,config_dir=config_dir,fusion_method=fusion_method)
    
    source_dir = os.path.join(npy_path,'ab3dmot_detection')
    
    destination_dir = f'./DMSTrack/AB3DMOT/data/v2v4real/detection/cobevt_Car_val'
    
    for i in range(9):
        src_txt_path = os.path.join(source_dir,f'000{i}.txt')
        des_txt_path = os.path.join(destination_dir,f'000{i}.txt')
        copy_txt_file(source_file=src_txt_path,target_file=des_txt_path)


    # copy_folder(src=source_txts, dst=destination_txts)
    
    args = cobevtmain.parse_args()
    cobevtmain.main(args)
    
    
    sys.argv.append('cobevt_Car_val_H1')
    sys.argv.append(1)
    sys.argv.append('3D')
    sys.argv.append(0.25)
    sys.argv.append('0000')
    sys.argv.append('val')

    for i in range(9):
        print(i)
        # evaluate.EVAL_SEQ = f'000{i}'
        # argv = ['cobevt_Car_val_H1',1,'3D',0.25,f'000{i}','val']
        sys.argv[5]=f'000{i}'
        print(sys.argv[5])
        evaluate.evaluate_main()

        source_file = os.path.join('./DMSTrack/AB3DMOT/results/v2v4real/cobevt_Car_val_H1','summary_car_average_eval3D.txt')
        dest_file = os.path.join(data_dir,f'summary_car_average_eval3D_{feature}_000{i}.txt')

        copy_txt_file(source_file, dest_file)




def gen_data_and_npy(
        config_dir = './V2X-ATCT/target_tracking/late_fusion',
        fusion_method = 'no_fusion_keep_all',
        only_test = False,
        time = 1,
        v2x_dataset_path = '',
        carnum=3,
        speed=60

):  
    if only_test == False:
        data_dir = gen_all_data(time=time,SAVE_PATH_ROOT='./V2X-ATCT/target_tracking/rq1/result/',v2x_dataset_path=v2x_dataset_path,carnum=carnum,speed=speed)

    trans_data_to_npy(data_dir=data_dir,config_dir=config_dir,fusion_method=fusion_method)
    return data_dir


    

def main(save_path_dir = './V2X-ATCT/target_tracking/rq1/result/',
         select_seed_num = 1,
         gen_seed_num = 5,
         insert_time = 3,
         v2x_dataset_path = '',
         speed=60,
         carnum=3
         ):

    now = datetime.now()
    formatted = now.strftime("%Y-%m-%d_%H:%M:%S")
    SAVE_PATH = os.path.join(save_path_dir,f'RQ1_EXP_{formatted}')

    v2x_dataset = v2x_dataset_path
    select_seeds = []
    for i in range(0,insert_time):
        
        data_list = []
        for index in range(0,gen_seed_num):
            data_dir = gen_data_and_npy(time=i+1,v2x_dataset_path=v2x_dataset,carnum=carnum,speed=speed)
            data_list.append(data_dir)


        min_fitness = 10
        min_fitness_path = ''
        for data_dir in data_list:
            data = rq2.read_yaml(os.path.join(data_dir, 'eval.yaml'))          
            ai50 = data['ave_iou_50']
            fitness = ai50
            if fitness < min_fitness :
                min_fitness = fitness
                min_fitness_path = data_dir



        select_seed_path = os.path.join(SAVE_PATH,f'select_seed_{i}_time')
        select_seeds.append(select_seed_path)
        copy_folder(src=min_fitness_path,dst=select_seed_path)

        if i == 2:
            continue

        npy_path = os.path.join(select_seed_path,'npy')
        if os.path.exists(npy_path):
            shutil.rmtree(npy_path)

        os.system(f'python datasetinit.py -d {select_seed_path}')
        v2x_dataset = os.path.join(select_seed_path,'v2x_dataset')

        
    for seed in select_seeds:
        trans_data_to_npy(data_dir=seed,config_dir='./V2X-ATCT/target_tracking/late_fusion',fusion_method= 'no_fusion_keep_all')
        MOT_operation(save_dir=seed,feature='fusion',seq_eval_mode='all')
        MOT_operation(save_dir=seed,feature='pos',seq_eval_mode='all')
        MOT_operation(save_dir=seed,feature='bev',seq_eval_mode='all')
        cal_cobevt_result(data_dir=seed)


    for index,seed in enumerate(select_seeds):
        fusion_amota = 0
        fusion_amotp = 0
        pos_amota = 0
        pos_amotp = 0
        bev_amota = 0
        bev_amotp = 0
        cobevt_amota = 0
        cobevt_amotp = 0
        for i in range(1,10):
            fusion_res = read_per_eva_result(dir_path=seed,time=f'{index+1}',file_name=f'summary_car_average_eval3D_fusion_000{i}.txt')
            pos_res = read_per_eva_result(dir_path=seed,time=f'{index+1}',file_name=f'summary_car_average_eval3D_pos_000{i}.txt')
            bev_res = read_per_eva_result(dir_path=seed,time=f'{index+1}',file_name=f'summary_car_average_eval3D_bev_000{i}.txt')
            cobevt_res = read_per_eva_result(dir_path=seed,time=f'{index+1}',file_name=f'summary_car_average_eval3D_fusion_000{i}.txt')
            fusion_amota += fusion_res['AMOTA']
            fusion_amotp += fusion_res['AMOTP']
            bev_amota += bev_res['AMOTA']
            bev_amotp += bev_res['AMOTP']
            pos_amota += pos_res['AMOTA']
            pos_amotp += pos_res['AMOTP']
            cobevt_amota += cobevt_res['AMOTA']
            cobevt_amotp += cobevt_res['AMOTP']

        print(f'insert_times: {index+1}')
        print(f'system: fusion; AMOTA: {fusion_amota/9}; AMOTP: {fusion_amotp/9}')
        print(f'system: pos; AMOTA: {pos_amota/9}; AMOTP: {pos_amotp/9}')
        print(f'system: bev; AMOTA: {bev_amota/9}; AMOTP: {bev_amotp/9}')
        print(f'system: cobevt; AMOTA: {cobevt_amota/9}; AMOTP: {cobevt_amotp/9}')
        print('-------------------------------------------')

def args_parser():
    parser = argparse.ArgumentParser(description="rq1 command")
    parser.add_argument('--save_path_dir', type=str, required=True,
                        )
    parser.add_argument('--gen_seed_num', type=int, required=True,
                        )
    parser.add_argument('--insert_time', type=int, required=True,
                        )
    parser.add_argument('--system', type=str, required=False,
                        )
    parser.add_argument('--select_seed_num', type=int, required=False,
                        )
    parser.add_argument('--driving_behaviour', type=str, required=False,
                        )
    parser.add_argument('--speed', type=int, required=False,
                        )
    parser.add_argument('--carnum', type=int, required=False,
                        )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    opt = args_parser()
    save_path_dir = opt.save_path_dir 
    save_path_dir = os.path.dirname(save_path_dir)

    v2x_dataset_path = opt.v2x_dataset_path 
    v2x_dataset_path = os.path.dirname(v2x_dataset_path)

    gen_seed_num = opt.gen_seed_num
    insert_time = opt.insert_time

    speed = opt.speed
    carnum = opt.carnum


    main(save_path_dir=save_path_dir,v2x_dataset_path=v2x_dataset_path,gen_seed_num=gen_seed_num,insert_time=insert_time,speed=speed,carnum=carnum)
