import os , sys

import copy
import random
import argparse
import obj_transformation as trans
from logger import CLogger
from config.config import Config
from utils.v2x_object import V2XInfo


def rq1_vis_parser():
    parser = argparse.ArgumentParser(description="rq1 command")
    parser.add_argument('-t', '--transform',
                        help="select a transform operation, insert/delete/translation/scaling/rotation")
    parser.add_argument('-s', '--scene', help="the scene of dataset", default=1)
    args = parser.parse_args()
    return args


def rq1_vis(transformation='insert', scene=1):
    """
    Transform visual of every scene (V2XGen + baseline)

    :param transformation: insert, delete, scale, translation and rotation
    :param scene: 1-9 scenes
    :return:
    """
    dataset_config = Config(scene=scene)
    begin_index = dataset_config.begin_index
    data_num = dataset_config.scene_data_num

    select_data_num = dataset_config.select_data_num#总pcd数量

    # for bg_index in range(1, select_data_num + 1):#执行操作主方法
    #     if bg_index < begin_index:
    #         continue
    #     elif bg_index > data_num + begin_index:
    #         break
    for bg_index in range(1,2):
        CLogger.info(f"Background {bg_index}")

        # load vehicle info
        ego_info = V2XInfo(bg_index, dataset_config=dataset_config)
        cp_info = V2XInfo(bg_index, is_ego=False, dataset_config=dataset_config)
        ego_info_baseline = copy.deepcopy(ego_info)
        cp_info_baseline = copy.deepcopy(cp_info)

        # random select car
        car_id = random.choice(list(ego_info.vehicles_info.keys()))

        if transformation == 'insert':#查看自此
            # same background as baseline
            trans.vehicle_insert(ego_info, cp_info, ego_info_baseline, cp_info_baseline)

            
        elif transformation == 'delete':
            # same bg data and car_id as baseline
            trans.vehicle_delete(ego_info, cp_info, ego_info_baseline, cp_info_baseline, car_id)
        elif transformation == 'translation':
            # same bg data, car_id and translation param as baseline
            trans.vehicle_translation(ego_info, cp_info, ego_info_baseline, cp_info_baseline, car_id)
        elif transformation == 'scaling':
            # same bg data, car_id and scaling rate as baseline
            trans.vehicle_scaling(ego_info, cp_info, ego_info_baseline, cp_info_baseline, car_id)
        elif transformation == 'rotation':
            # same bg data, car_id and rot degree as baseline
            trans.vehicle_rotation(ego_info, cp_info, ego_info_baseline, cp_info_baseline, car_id)


if __name__ == '__main__':
    cmd_args = rq1_vis_parser()
    rq1_vis(cmd_args.transform, int(cmd_args.scene))
