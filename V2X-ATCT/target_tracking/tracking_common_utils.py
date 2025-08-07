import pickle
import os , sys
# sys.path.append("/home/maji/Downloads/V2XGen-main/")
import yaml
import math
import numpy as np
from scipy.interpolate import interp1d
import utils.v2X_file as vf
import numpy as np
# from z3 import Real, Solver, sat
import open3d as o3d
import target_tracking.z3_solver as zs
import utils.common_utils as cu
import core.obj_insert as oi
import random
import target_tracking.scene_pcd_world_coordinate.gen_road_edge_points as grep


import config
from utils.common_utils import get_corners, center_system_transform, rz_degree_system_transform, pc_numpy_2_o3d
from core.lidar_simulation import lidar_simulation, lidar_intensity_convert
from build.mtest.utils.Utils_o3d import load_normalized_mesh_obj
from build.mtest.utils.Utils_common import get_geometric_info, change_3dbox, get_initial_box3d_in_bg
from build.mtest.core.occusion_handing.combine_pc import combine_single_pcd
from build.mtest.core.pose_estimulation.pose_generator import tranform_mesh_by_pose, generate_pose







def cal_x1y1_2_x2y2_distance(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)  
 


def move_to_point(dx_val=0.8,dy_val=0.8,tar_pos=[],ins_pos=[]):
    x_dis = tar_pos[0] - ins_pos[0]
    y_dis = tar_pos[1] - ins_pos[1]

    if x_dis > 0:
        dx=dx_val
    elif x_dis == 0:
        dx=0
    else:
        dx=-dx_val

    if y_dis > 0:
        dy=dy_val
    elif y_dis == 0:
        dy=0
    else:
        dy=-dy_val

    if abs(x_dis) < 0.2*dx_val:
        dx = x_dis
    if abs(y_dis) < 0.2*dy_val:
        dy = y_dis

    if abs(dx)==dx_val and abs(dy)!=dy_val and abs(x_dis)>dx_val:
        dx=1.4*dx
    if abs(dy)==dy_val and abs(dx)!=dx_val and abs(y_dis)>dy_val:
        dy=1.4*dy

    return dx,dy



def cal_velocity_through_coordinates(pos1,pos2):
    return math.sqrt((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2+(pos1[2]-pos2[2])**2)/0.1


def follow_car_action(ego_info={},cp_info={},insert_info={},followed_car='ego',init_insert=False,
                      MinFD=6,
                      MaxFD=18,
                      car_bf_l=4,
                      car_ins_w = 2,
                      car_ins_l=4,
                      dx_val=0.3,
                      dy_val=0.3,
                    #   lane_w = 3.5
                      ):
    
    v2x_info = ego_info if followed_car=='ego' else cp_info

    if init_insert == True:
        
        x = -np.random.uniform(MinFD+car_bf_l/2+car_ins_l/2,MaxFD+car_bf_l/2+car_ins_l/2)

        y = 0
        # y_list = [-car_ins_w,0,car_ins_w]
        # y = random.choice(y_list) 

        z = oi.select_road_height(v2x_info.road_pc,[x,y])

        world_ins_pos = zs.ego_xyz_to_world_coordinate_require_z(x,y,z,v2x_info.param['lidar_pose'])
        world_carbf_pos = zs.ego_xyz_to_world_coordinate_require_z(0,0,0,v2x_info.param['lidar_pose'])

        position = [x,y]
        rz_degree = 0

        # tar_pos = [tar_pos_x[0],tar_pos_y[0]]

        insert_info={
            'insert_position':position,
            'rzdegree':rz_degree,
            # 'tar_pos':tar_pos,

            'world_ins_pos': world_ins_pos,
            'world_carbf_pos':world_carbf_pos,

        }
    else :
        position = insert_info['insert_position']

        # if insert_info['carins_v']==None :
        if 'carins_v' not in insert_info :
            last_world_ins_pos = insert_info['world_ins_pos']
            last_world_carbf_pos = insert_info['world_carbf_pos']
            z = oi.select_road_height(v2x_info.road_pc,position)
            world_ins_pos = zs.ego_xyz_to_world_coordinate_require_z(position[0],position[1],z,v2x_info.param['lidar_pose'])
            world_carbf_pos = zs.ego_xyz_to_world_coordinate_require_z(0,0,0,v2x_info.param['lidar_pose'])

            insert_info['world_ins_pos']=world_ins_pos
            insert_info['world_carbf_pos']=world_carbf_pos

            insert_info['carins_v']=cal_velocity_through_coordinates(last_world_ins_pos,world_ins_pos)
            insert_info['carbf_v']=cal_velocity_through_coordinates(last_world_carbf_pos,world_carbf_pos)

        else:
            last_carins_v = insert_info['carins_v']
            last_carbf_v = insert_info['carbf_v']
            last_world_ins_pos = insert_info['world_ins_pos']
            last_world_carbf_pos = insert_info['world_carbf_pos']

            if last_carins_v > last_carbf_v:
                dx = dx_val
                # dy = dy_val
            elif last_carins_v == last_carbf_v :
                dx = 0
                # dy = 0

            else :
                dx = -dx_val

            new_x = position[0]+dx
            new_y = 0
            
            if new_x <= -(MinFD+car_bf_l/2+car_ins_l/2) and new_x >= -(MaxFD+car_bf_l/2+car_ins_l/2) :
                print('ok')
                insert_info['insert_position'] = [new_x,new_y]
            
            position = insert_info['insert_position']
            z = oi.select_road_height(v2x_info.road_pc,position)
            world_ins_pos = zs.ego_xyz_to_world_coordinate_require_z(position[0],position[1],z,v2x_info.param['lidar_pose'])
            world_carbf_pos = zs.ego_xyz_to_world_coordinate_require_z(0,0,0,v2x_info.param['lidar_pose'])

            insert_info['world_ins_pos']=world_ins_pos
            insert_info['world_carbf_pos']=world_carbf_pos

            insert_info['carins_v']=cal_velocity_through_coordinates(last_world_ins_pos,world_ins_pos)
            insert_info['carbf_v']=cal_velocity_through_coordinates(last_world_carbf_pos,world_carbf_pos)
            






    if followed_car=='cp':

        x = insert_info['insert_position'][0]
        y = insert_info['insert_position'][1]
        z = oi.select_road_height(cp_info.road_pc,[x,y])
        rz_degree = insert_info['rzdegree']
        k = math.tan(rz_degree)
        b = y - k*x

        x1 = x+1
        y1 = x1*k + b





        x_transformed,y_transformed = list(cu.center_system_transform([x,y,z], cp_info.param['lidar_pose'], ego_info.param['lidar_pose']))[:2]
        x1_transformed,y1_transformed = list(cu.center_system_transform([x1,y1,z], cp_info.param['lidar_pose'], ego_info.param['lidar_pose']))[:2]
        transformed_rz_degree = cal_angle_with_tan(-(y_transformed-y1_transformed)/(x_transformed-x1_transformed))

        insert_info['transformed_position']=[x_transformed,y_transformed]
        insert_info['transformed_rzdegree']=transformed_rz_degree


    

    return insert_info





def overtake_action(ego_info={},cp_info={},insert_info={},followed_car='ego',init_insert=False,
                        direction='right',
                        init_ins_area=[-5,-7,-0.5,0.5],
                        carbo_l = 4,
                        carins_l = 4,
                        lane_w = 3.5,
                        MinDF = 8,
                        MaxDF = 16,
                        dx_val = 0.3,
                        dy_val = 0.3,
                      ):

    


    if init_insert==True:
        max_limit = carbo_l/2 + carins_l/2 + MaxDF 
        min_limit = carbo_l/2 + carins_l/2 + MinDF
        assert min_limit > (carbo_l/2+carins_l/2+lane_w )

        
        x1 = np.random.uniform(-max_limit,-min_limit)
        y1 = 0
        p1 = [x1,y1]

        x2 = np.random.uniform(x1,-(carbo_l/2+carins_l/2))
        y2 = lane_w

        p2 = [x2,y2]


        x4 = np.random.uniform(min_limit,max_limit)
        y4 = 0

        p4 = [x4,y4]

        x3 = np.random.uniform((carbo_l/2+carins_l/2),x4)
        y3 = lane_w
        p3 = [x3,y3]    

        if direction=='left':
            p2[1] = -p2[1]   
            p3[1] = -p3[1]                                                                     
        
        insert_position = [np.random.uniform(-max_limit,x1),0]
        rz_degree = 0

        
        insert_info={'insert_position':insert_position,
                     'rzdegree':rz_degree,
                     'p_list':[p1,p2,p3,p4],
                     'tar_pos_index':0,        
        }




    else:
        insert_position = insert_info['insert_position']
        tar_pos_index = insert_info['tar_pos_index']
        tar_pos = insert_info['p_list'][tar_pos_index]
        rz_degree = insert_info['rzdegree']
        dis = cal_x1y1_2_x2y2_distance(insert_position[0],insert_position[1],tar_pos[0],tar_pos[1])



        if dis < 0.5 or insert_position[0] > tar_pos[0]:

            if tar_pos_index < len(insert_info['p_list'])-1:
                tar_pos_index = tar_pos_index+1
                insert_info['tar_pos_index'] = tar_pos_index
                dx,dy = move_to_point(dx_val=dx_val,dy_val=dy_val,tar_pos=tar_pos,ins_pos=insert_position)
                insert_position[0]=insert_position[0]+dx
                insert_position[1]=insert_position[1]+dy

            else:
                insert_info['insert_position'] = [insert_info['insert_position'][0]+0.2,insert_info['insert_position'][1]]
            
            
        else:
            dx,dy = move_to_point(dx_val=dx_val,dy_val=dy_val,tar_pos=tar_pos,ins_pos=insert_position)
            insert_position[0]=insert_position[0]+dx
            insert_position[1]=insert_position[1]+dy




    if followed_car=='cp':

        x = insert_info['insert_position'][0]
        y = insert_info['insert_position'][1]
        z = oi.select_road_height(cp_info.road_pc,[x,y])
        rz_degree = insert_info['rzdegree']
        k = math.tan(rz_degree)
        b = y - k*x

        x1 = x+1
        y1 = x1*k + b





        x_transformed,y_transformed = list(cu.center_system_transform([x,y,z], cp_info.param['lidar_pose'], ego_info.param['lidar_pose']))[:2]
        x1_transformed,y1_transformed = list(cu.center_system_transform([x1,y1,z], cp_info.param['lidar_pose'], ego_info.param['lidar_pose']))[:2]
        transformed_rz_degree = cal_angle_with_tan(-(y_transformed-y1_transformed)/(x_transformed-x1_transformed))

        insert_info['transformed_position']=[x_transformed,y_transformed]
        insert_info['transformed_rzdegree']=transformed_rz_degree


    return insert_info







def cal_angle_with_tan(tan_value):

    radian = math.atan(tan_value)
    degree = math.degrees(radian)

    return degree




def avoid_collision2(ego_info={},
                    cp_info={},
                    insert_position=[],
                    
                    ):

    half_ins_car_length = 2.4/2
    half_ins_car_width = 1.06/2
    pos_change_flag = False

    vehicles = {}
    ego_vehicles = ego_info.vehicles_info
    cp_vehicles = cp_info.vehicles_info

    vehicles['ego']={'center':[0,0,0],'lwh':[2.4*2,1.1*2,1.2*2]}

    for id,val in ego_vehicles.items():
        vehicles[f'{id}']={'center':val['center'],'lwh':[val['length'],val['width'],val['height']]}

    pos = cu.center_system_transform([0,0,0],cp_info.param['lidar_pose'],ego_info.param['lidar_pose'])

    vehicles['cp']={'center':pos,'lwh':[2.2*2,1*2,1.1*2]}

    for id,val in cp_vehicles.items():
        if vehicles.get(f'{id}') is None:
            pos = cu.center_system_transform(val['center'],cp_info.param['lidar_pose'],ego_info.param['lidar_pose'])
            vehicles[f'{id}']={'center':pos,'lwh':[val['length'],val['width'],val['height']]}



    for id,val in vehicles.items():
        x_dis,y_dis = val['center'][0]-insert_position[0],val['center'][1]-insert_position[1]
        x_limit,y_limit = val['lwh'][0]/4 - half_ins_car_length,val['lwh'][1]/4 - half_ins_car_width


        if abs(x_dis) < abs(x_limit) or abs(y_dis) < abs(y_limit):
            if pos_change_flag == False:
                pos_change_flag = True

            for i in range(0,2): 
                if  abs(x_dis) < abs(x_limit):
                    insert_position=avoid_collision_action2(xy=insert_position,vehicle=[val['center'][0],val['center'][1]],dx=0.4,dy=0.4,change_direction='x')
                
                if  abs(y_dis) < abs(y_limit):
                    insert_position=avoid_collision_action2(xy=insert_position,vehicle=[val['center'][0],val['center'][1]],dx=0.4,dy=0.4,change_direction='y')
                
                
                x_dis,y_dis = val['center'][0]-insert_position[0],val['center'][1]-insert_position[1]
                if abs(x_dis) >= abs(x_limit) and abs(y_dis) >= abs(y_limit):
                    break
            

        

    return pos_change_flag, insert_position



def avoid_collision_action2(xy,vehicle,dx,dy,change_direction='x'):
    if change_direction=='x':
        if xy[0] > vehicle[0]:
            xy[0] =  xy[0]+dx
        else:
            xy[0] = xy[0] -dx
        print('+x')
    else:
        if xy[1] > vehicle[1]:
            xy[1] =  xy[1]+dy
        else:
            xy[1] = xy[1] -dy
        print('+y')
    print("avoid!!")
    return xy




def avoid_collision(ego_info={},
                    cp_info={},
                    insert_position=[],
                    objs_index=3,
                    objs_index_require=False
                    ):
    
    pos_change_flag = False

    
    if objs_index_require == True :
        obj_filename = config.common_config.obj_filename
        assets_dir = config.common_config.obj_dir_path
        obj_car_dirs = os.listdir(config.common_config.obj_dir_path)
        objs_index = objs_index
        obj_mesh_path = os.path.join(assets_dir, obj_car_dirs[objs_index], obj_filename)
        # print(obj_mesh_path)
        mesh_obj_initial = oi.load_normalized_mesh_obj(obj_mesh_path)
        half_diagonal, center, half_height = get_geometric_info(mesh_obj_initial)

    else:
        half_diagonal=2.62092847379101

    vehicles = {}
    ego_vehicles = ego_info.vehicles_info
    cp_vehicles = cp_info.vehicles_info

    vehicles['ego']={'center':[0,0,0],'lwh':[2.4,1.1,1.2]}

    for id,val in ego_vehicles.items():
        vehicles[f'{id}']={'center':val['center'],'lwh':[val['length'],val['width'],val['height']]}

    pos = cu.center_system_transform([0,0,0],cp_info.param['lidar_pose'],ego_info.param['lidar_pose'])

    vehicles['cp']={'center':pos,'lwh':[2.2,1,1.1]}

    for id,val in cp_vehicles.items():
        if vehicles.get(f'{id}') is None:
            pos = cu.center_system_transform(val['center'],cp_info.param['lidar_pose'],ego_info.param['lidar_pose'])
            vehicles[f'{id}']={'center':pos,'lwh':[val['length'],val['width'],val['height']]}


    for id,val in vehicles.items():
        x_dis,y_dis = val['center'][0]-insert_position[0],val['center'][1]-insert_position[1]
        dis = math.sqrt(x_dis ** 2 + y_dis ** 2)
        if dis < 3:
            if pos_change_flag == False:
                pos_change_flag = True
            insert_position=avoid_collision_action(xy=insert_position,vehicle=[val['center'][0],val['center'][1]],dx=0.4,dy=0.4)

    return pos_change_flag, insert_position

def avoid_collision_action(xy,vehicle,dx,dy):

    if xy[0] > vehicle[0]:
        xy[0] =  xy[0]+dx
    else:
        xy[0] = xy[0] -dx


    if xy[1] > vehicle[1]:
        xy[1] =  xy[1]+dy
    else:
        xy[1] = xy[1] -dy

    print("avoid!!")
    return xy


def collision_detection(xy,vehicles,half_diagonal):

    for id,val in vehicles.items():
        x_dis,y_dis = val['center'][0]-xy[0],val['center'][1]-xy[1]
        dis = math.sqrt(x_dis ** 2 + y_dis ** 2)

        diagonal = math.sqrt(val['lwh'][0]**2+val['lwh'][1]**2)/2

        length = diagonal + half_diagonal

        length = length*0.7

        if dis >= length:
            pass
        else:
            return True
    return False
   


def read_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                lines = [line.strip() for line in lines]
                return lines
    except FileNotFoundError:
            print("error: file not found!")
    except Exception as e:
            print(f"error: unknown error: {e}")
    return []

def get_points(base_path,path_list):
    points = []
    for path in path_list.split(','):
        if '.txt' not in path:
            continue

        load_path = os.path.join(base_path,path.strip())
        res = read_txt(load_path)

        for point in res:
            point = point.split(',')
            points.append([float(point[1]),float(point[2]),float(point[3])])

    return points


def gen_init_tar_pos(points,
                     lane_w=3.75,
                     car_w=1.4,
                    #  distance_range=[0.5,1.5],
                    lane_num=2,
                     ):
    new_points = []

    distance = np.random.uniform(car_w/2,lane_num*lane_w-car_w/2)

    for index,point in enumerate(points):
        if index == 0:
            k_v,b_v = cal_vertical_slope_and_b(points[1][0],points[1][1],point[0],point[1])
            if k_v == False :
                points[1][1] = points[1][1] +0.001
                point[1] = point[1] -0.001
                k_v,b_v = cal_vertical_slope_and_b(points[1][0],points[1][1],point[0],point[1])

            
            new_pont = zs.find_point_on_line(B=point,C=points[1],k=k_v,b=b_v,n=distance)
            new_points.append([new_pont[0],new_pont[1],point[2]])

        elif index == len(points)-1 :
            k_v,b_v = cal_vertical_slope_and_b(points[index-1][0],points[index-1][1],points[index][0],points[index][1])

            if k_v == False :
                points[index-1][1] = points[index-1][1] + 0.001
                points[index][1] = points[index][1] -0.001
                k_v,b_v = cal_vertical_slope_and_b(points[index-1][0],points[index-1][1],points[index][0],points[index][1])

            k,b = cal_slope_and_b(points[index-1][0],points[index-1][1],points[index][0],points[index][1])
            if k == False :
                points[index-1][0] = points[index-1][0] +0.001
                points[index][0] = points[index][0] - 0.001
                k,b = cal_slope_and_b(points[index-1][0],points[index-1][1],points[index][0],points[index][1])

            C = zs.find_point_one_line_2(k=k,b=b,point=points[index-1],oripoint=points[index])

            new_pont = zs.find_point_on_line(B=point,C=C,k=k_v,b=b_v,n=distance)
            new_points.append([new_pont[0],new_pont[1],point[2]])
            
        else :
            k_v,b_v = cal_vertical_slope_and_b(points[index-1][0],points[index-1][1],points[index][0],points[index][1])
            if k_v == False :
                points[index-1][1] = points[index-1][1] + 0.001
                points[index][1] = points[index][1] -0.001
                k_v,b_v = cal_vertical_slope_and_b(points[index-1][0],points[index-1][1],points[index][0],points[index][1])

            new_pont = zs.find_point_on_line(B=point,C=points[index+1],k=k_v,b=b_v,n=distance)
            new_points.append([new_pont[0],new_pont[1],point[2]])

        

    return new_points,distance
def gen_init_tar_pos_old(points,
                     lane_w=3.75,
                     car_w=1.4,
                    #  distance_range=[0.5,1.5],
                    lane_num=2,
                     ):
    new_points = []



    for index,point in enumerate(points):
        if index == 0:
            k_v,b_v = cal_vertical_slope_and_b(points[1][0],points[1][1],point[0],point[1])
            if k_v == False :
                points[1][1] = points[1][1] +0.001
                point[1] = point[1] -0.001
                k_v,b_v = cal_vertical_slope_and_b(points[1][0],points[1][1],point[0],point[1])

            distance = np.random.uniform(car_w/2,lane_num*lane_w-car_w/2)
            new_pont = zs.find_point_on_line(B=point,C=points[1],k=k_v,b=b_v,n=distance)
            new_points.append([new_pont[0],new_pont[1],point[2]])

        elif index == len(points)-1 :
            k_v,b_v = cal_vertical_slope_and_b(points[index-1][0],points[index-1][1],points[index][0],points[index][1])

            if k_v == False :
                points[index-1][1] = points[index-1][1] + 0.001
                points[index][1] = points[index][1] -0.001
                k_v,b_v = cal_vertical_slope_and_b(points[index-1][0],points[index-1][1],points[index][0],points[index][1])

            k,b = cal_slope_and_b(points[index-1][0],points[index-1][1],points[index][0],points[index][1])
            if k == False :
                points[index-1][0] = points[index-1][0] +0.001
                points[index][0] = points[index][0] - 0.001
                k,b = cal_slope_and_b(points[index-1][0],points[index-1][1],points[index][0],points[index][1])

            C = zs.find_point_one_line_2(k=k,b=b,point=points[index-1],oripoint=points[index])

            distance = np.random.uniform(car_w/2,lane_num*lane_w-car_w/2)
            new_pont = zs.find_point_on_line(B=point,C=C,k=k_v,b=b_v,n=distance)
            new_points.append([new_pont[0],new_pont[1],point[2]])
            
        else :
            k_v,b_v = cal_vertical_slope_and_b(points[index-1][0],points[index-1][1],points[index][0],points[index][1])
            if k_v == False :
                points[index-1][1] = points[index-1][1] + 0.001
                points[index][1] = points[index][1] -0.001
                k_v,b_v = cal_vertical_slope_and_b(points[index-1][0],points[index-1][1],points[index][0],points[index][1])

            distance = np.random.uniform(car_w/2,lane_num*lane_w-car_w/2)
            new_pont = zs.find_point_on_line(B=point,C=points[index+1],k=k_v,b=b_v,n=distance)
            new_points.append([new_pont[0],new_pont[1],point[2]])

        

    return new_points



def cal_slope_and_b(x1,y1,x2,y2):

    if x1 == x2:
        return False,x1
    elif y1 == y2 :
        return 0.000001,y1
    else:
        k=(y2-y1)/(x2-x1)
        b = y2-k*x2
        return k,b


def cal_vertical_slope_and_b(x1,y1,x2,y2):

    if x1 == x2:
        return 0.000001,y2
    elif y1 == y2 :
        return False,x2

    else:
        k=(y2-y1)/(x2-x1)
        vertical_k = -1/k
        b = y2-vertical_k*x2
        return vertical_k,b
    
    return



def define_reaching_condition(k,b,ins_pos):
    print(ins_pos)
    y = k*ins_pos[0] + b
    ins_pos_y = ins_pos[1]

    if ins_pos_y < y:
        return 'greater_than'
    else :
        return 'less_than'    


def point_to_line_distance(point, k, b):
    x0 = point[0]
    y0 = point[1]
    A = k
    B = -1
    C = b
    numerator = abs(A * x0 + B * y0 + C)
    denominator = math.sqrt(A ** 2 + B ** 2)
    return numerator / denominator

#
def truning_action( 
                    # reaching_threshold=0.3,
                   points=[],
                   init_insert=False,
                   ego_info={},
                   cp_info={},
                   insert_info={},
                   dx_val = 0.8,
                   dy_val = 0.8,
                   start_point_index=0,
                   ):
    reaching_threshold = math.sqrt(dx_val**2+dy_val**2)*1.01

    if init_insert:
        x,y = zs.world_to_ego_coordinate(points[start_point_index][0],points[start_point_index][1],points[start_point_index][2],ego_info.param['lidar_pose'])
        
        ins_pos_world = [points[start_point_index][0],points[start_point_index][1],points[start_point_index][2]]#init insert pos

        tar_pos_index = start_point_index+1# cal slope and b
        last_tar_pos_index = start_point_index

        tar_pos=points[tar_pos_index]
        last_tar_pos=points[last_tar_pos_index]

        k,b = cal_vertical_slope_and_b(last_tar_pos[0],last_tar_pos[1],tar_pos[0],tar_pos[1])

        if k == False:
            last_tar_pos[1] = last_tar_pos[1] + 0.001
            tar_pos[1] = tar_pos[1] - 0.001
            k,b = cal_vertical_slope_and_b(last_tar_pos[0],last_tar_pos[1],tar_pos[0],tar_pos[1])
        
        drc = define_reaching_condition(k,b,ins_pos_world)
        
        tar_pos_ego = zs.world_to_ego_coordinate(tar_pos[0],tar_pos[1],tar_pos[2],ego_info.param['lidar_pose'])
        last_tar_pos_ego = zs.world_to_ego_coordinate(last_tar_pos[0],last_tar_pos[1],last_tar_pos[2],ego_info.param['lidar_pose'])

        
        if tar_pos_ego[0] == last_tar_pos_ego[0]:
            last_tar_pos_ego[0] = last_tar_pos_ego[0] + 0.001
            tar_pos_ego[0] = tar_pos_ego[0] - 0.001
        rz_degree = cal_angle_with_tan(-(tar_pos_ego[1]-last_tar_pos_ego[1])/(tar_pos_ego[0]-last_tar_pos_ego[0]))


        insert_info={
            'insert_position':[x,y],
            'ins_pos_world':ins_pos_world,
            'tar_pos_index':tar_pos_index,
            'last_tar_pos_index':last_tar_pos_index,
            'pos_len':len(points),
            'k_b':[k,b],
            'drc':drc,
            'rz_degree': rz_degree,         
        }


        return insert_info

    else:
        
        tar_pos_index = insert_info['tar_pos_index']
        tar_pos = points[tar_pos_index]
        ins_pos = insert_info['insert_position']
        z = oi.select_road_height(ego_info.road_pc,ins_pos)

        world_ins_pos = insert_info['ins_pos_world']
        last_tar_pos_index = insert_info['last_tar_pos_index']
        last_tar_pos=points[last_tar_pos_index]


        k,b  = insert_info['k_b']
        drc = insert_info['drc']

        reach_y = world_ins_pos[0]*k+b

        dis = point_to_line_distance(world_ins_pos,k,b)

        if world_ins_pos[1] >= reach_y or dis <reaching_threshold if drc == 'greater_than' else world_ins_pos[1] <= reach_y or dis <reaching_threshold :

            if tar_pos_index < insert_info['pos_len']-1:
                last_tar_pos_index = tar_pos_index
                tar_pos_index = tar_pos_index+1
                
                last_tar_pos = points[last_tar_pos_index]
                tar_pos = points[tar_pos_index]

                dx,dy = move_to_tar_pos(dx_val=dx_val,dy_val=dy_val,tar_pos=tar_pos,ins_pos=world_ins_pos,reaching_flag=False)
                
                world_ins_pos[0] = world_ins_pos[0] + dx
                world_ins_pos[1] = world_ins_pos[1] + dy

                x,y = zs.world_to_ego_coordinate(world_ins_pos[0],world_ins_pos[1],world_ins_pos[2],ego_info.param['lidar_pose'])

                k,b = cal_vertical_slope_and_b(last_tar_pos[0],last_tar_pos[1],tar_pos[0],tar_pos[1])

                if k == False: 
                    last_tar_pos[1] = last_tar_pos[1] + 0.001
                    tar_pos[1] = tar_pos[1] - 0.001
                    k,b = cal_vertical_slope_and_b(last_tar_pos[0],last_tar_pos[1],tar_pos[0],tar_pos[1])

                drc = define_reaching_condition(k,b,last_tar_pos)


                tar_pos_ego = zs.world_to_ego_coordinate(tar_pos[0],tar_pos[1],tar_pos[2],ego_info.param['lidar_pose'])
                last_tar_pos_ego = zs.world_to_ego_coordinate(last_tar_pos[0],last_tar_pos[1],last_tar_pos[2],ego_info.param['lidar_pose'])

                if tar_pos_ego[0] == last_tar_pos_ego[0]:
                    last_tar_pos_ego[0] = last_tar_pos_ego[0] + 0.001
                    tar_pos_ego[0] = tar_pos_ego[0] - 0.001

                rz_degree = cal_angle_with_tan(-(tar_pos_ego[1]-last_tar_pos_ego[1])/(tar_pos_ego[0]-last_tar_pos_ego[0]))

                insert_info={
                    'insert_position':[x,y],
                    'ins_pos_world':[world_ins_pos[0],world_ins_pos[1],world_ins_pos[2]],
                    'tar_pos_index':tar_pos_index,
                    'last_tar_pos_index':last_tar_pos_index,
                    
                    'pos_len':len(points),
                    'k_b':[k,b],
                    'drc':drc,
                    'rz_degree': rz_degree,

                }

                
                return insert_info
            
            else :
                world_ins_pos = insert_info['ins_pos_world']
                x,y = zs.world_to_ego_coordinate(world_ins_pos[0],world_ins_pos[1],world_ins_pos[2],ego_info.param['lidar_pose'])
                insert_info['insert_position']  = [x,y]
                return insert_info


        else :

            dx,dy = move_to_tar_pos(dx_val=dx_val,dy_val=dy_val,tar_pos=tar_pos,ins_pos=world_ins_pos,reaching_flag=False)
            
            world_ins_pos[0] = world_ins_pos[0] + dx
            world_ins_pos[1] = world_ins_pos[1] + dy
            x,y = zs.world_to_ego_coordinate(world_ins_pos[0],world_ins_pos[1],world_ins_pos[2],ego_info.param['lidar_pose'])
            insert_info={
                'insert_position':[x,y],
                'ins_pos_world':[world_ins_pos[0],world_ins_pos[1],world_ins_pos[2]],
                'tar_pos_index':tar_pos_index,
                'last_tar_pos_index':last_tar_pos_index,

                'pos_len':len(points),
                'k_b':[k,b],
                'drc':drc,
                'rz_degree': insert_info['rz_degree'],
            }
           

            return insert_info
        

        return False




def move_to_tar_pos(tar_pos,ins_pos,dx_val=0.8,dy_val=0.8,
                    reaching_flag=False,
                    reaching_threshold=1.5, #
                    

                    ):  

    x_dis = tar_pos[0] - ins_pos[0]
    if x_dis > 0 :
        if reaching_flag:
            dx = dx_val/3
        else:
            dx = dx_val
    else :
        if reaching_flag:
            dx = -dx_val/3
        else:
            dx = -dx_val

    y_dis = tar_pos[1] - ins_pos[1]
    if y_dis > 0 :
        if reaching_flag:
            dy = dy_val/3
        else:
            dy = dy_val
    else :
        if reaching_flag:
            dy = -dy_val/3
        else:
            dy = -dy_val
    
    xy_threshold = math.sqrt(reaching_threshold)
    if abs(x_dis) < xy_threshold:
        dx = dx/4
    elif abs(y_dis) < xy_threshold:
        dy = dy/4

    if abs(dx) == dx_val and abs(dy)!= dy_val:
        dx = 2*dx

    if abs(dx) != dx_val and abs(dy) == dy_val:
        dy = 2*dy

    return dx,dy


   


    

        
