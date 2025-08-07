# V2X-ATCT
Repository provides the code of V2X-ATCT project.

## Installation

Follow the steps below to complete the installation.

### Basice Dependency

Create conda environment, python >= 3.7.

```shell
conda create -n v2xatct python=3.7
conda activate v2xatct
```

Follow the steps in the README.md under the V2X-ATCT directory to complete the installation.

Follow the steps in the README.md under the DMSTrack directory to complete the installation.





## Quick Start

You can run the program to generate the scenario.

```shell
$ python V2X-ATCT/target_tracking/Generate_scenes.py --save_path ${save_path} --scene_num ${scene_num}
```

- scene_num in 1-9.
- save_path: The path you want to save.





## Experiments
### RQ1

Run the following command to conduct the first experiment.

```shell
$ python V2X-ATCT/target_tracking/rq1/RQ1.py --gen_seed_num ${gen_seed_num} --insert_time ${insert_time} --v2x_dataset_path ${v2x_dataset_path}
```
- gen_seed_num: The number of seeds generated per round. Default 5.
- insert_time: The number of inserted trajectories. Default 3.
- v2x_dataset_path: The path of the original dataset after initialization.




### RQ2

Run the following command to conduct the second experiment.

```shell
$ V2X-ATCT/target_tracking/rq2/RQ2.py --seed_num ${seed_num} 
```
- seed_num: The number of seeds generated per round. Default 5.
