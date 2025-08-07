import os
import utils.v2X_file as v2x

# v2x_dataset_root = f"/home/software/V2V4Real_semantic/v2x_dataset"
v2x_dataset_root = f"/home/maji/Downloads/V2X-ATCT/v2x_dataset"
dataset = "rq1"     # default dataset

# ego
ego_road_split_pc_dir = f"{v2x_dataset_root}/{dataset}/0/road_pcd"
ego_road_split_label_dir = f"{v2x_dataset_root}/{dataset}/0/predictions"
ego_pc_dir = f"{v2x_dataset_root}/{dataset}/0/velodyne"
ego_label_dir = f"{v2x_dataset_root}/{dataset}/0/labels"

# 协同
coop_road_split_pc_dir = f"{v2x_dataset_root}/{dataset}/1/road_pcd"
coop_road_split_label_dir = f"{v2x_dataset_root}/{dataset}/1/predictions"
coop_pc_dir = f"{v2x_dataset_root}/{dataset}/1/velodyne"
coop_label_dir = f"{v2x_dataset_root}/{dataset}/1/labels"

v2x_dataset_saved_dir = f"{v2x_dataset_root}/{dataset}/"

# rq requirement
select_data_num = 200
