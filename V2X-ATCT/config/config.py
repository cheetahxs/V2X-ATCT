import os
import yaml


class Config:
    """
    Dataset config
    """
    def __init__(self, dataset="v2x_dataset", scene=1,config_path="V2X-ATCT/config/dataset_config.yml"):
        with open(config_path, "r") as config_file:
        # with open("/home/maji/Downloads/V2XTargetTracking-main/V2XGen-main/config/dataset_config.yml", "r") as config_file:
            dataset_config = yaml.load(config_file, Loader=yaml.FullLoader)

        self.dataset_root = dataset_config.get("dataset_path", "/${your dataset path}")
        print(self.dataset_root)
        self.select_data_num = 1993#场景总帧数
        self.dataset = dataset

        # V2V4Real test dataset
        if dataset == "v2x_dataset":
            # select data scene
            index_dict = {
                1: 147,
                2: 114,
                3: 144,
                4: 198,
                5: 180,
                6: 310,
                7: 304,
                8: 221,
                9: 375
            }
            begin = 1
            for i, v in index_dict.items():
                if i < scene:
                    begin += v
            self.begin_index = begin#获取某个场景的索引
            # self.scene_data_num = index_dict[scene] - 1#该场景的总数
            self.scene_data_num = index_dict[scene] #该场景的总数
        # rq1 dataset
        elif dataset == "rq1":
            # 200 random data
            dataset = "rq1/random_200"
            self.select_data_num = 200
        # rq_eval data generate
        elif dataset == "rq_eval":
            dataset = "rq2/pre_trans_dataset"
            # dataset = "rq3/test_dataset"
        self.dataset_path = os.path.join(self.dataset_root, dataset)#路径结合

        # data path
        # ego
        self.ego_road_split_pc_dir = f"{self.dataset_path}/0/road_pcd"
        self.ego_road_split_label_dir = f"{self.dataset_path}/0/predictions"
        self.ego_pc_dir = f"{self.dataset_path}/0/pcd"
        self.ego_label_dir = f"{self.dataset_path}/0/yaml"

        # cooperative
        self.coop_road_split_pc_dir = f"{self.dataset_path}/1/road_pcd"
        self.coop_road_split_label_dir = f"{self.dataset_path}/1/predictions"
        self.coop_pc_dir = f"{self.dataset_path}/1/pcd"
        self.coop_label_dir = f"{self.dataset_path}/1/yaml"

        self.v2x_dataset_saved_dir = f"{self.dataset_path}/"#data/v2x_dataset

        if self.dataset == "rq_eval":
            self.ego_pc_dir = f"{self.dataset_path}/0/pcd"
            self.coop_pc_dir = f"{self.dataset_path}/1/pcd"



