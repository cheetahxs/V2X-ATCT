import config.lidar_config
import config.common_config
import config.v2x_config
import os
import sys

modality = "multi"
not_behind_initial_obj = True
# os.environ['PROJECT_DIR'] ="/home/software/V2V4Real_semantic/MultiTest"
# os.environ['PROJECT_DIR'] = config.common_config.project_dir
os.environ['PROJECT_DIR'] = "/home/software/V2V4Real_semantic/MultiTest"
# sys.path.append(config.common_config.project_dir)
sys.path.append("/home/software/V2V4Real_semantic/MultiTest")

