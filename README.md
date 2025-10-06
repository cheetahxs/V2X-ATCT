# Start

This project was run on a configuration with Ubuntu 22.04 LTS, 32GB RAM, an i9-11900K processor, and an RTX 4070 Ti.


## **1. Download Data**
Clone or download all project files.
Download the dataset from the Google Drive link (https://drive.google.com/file/d/14dgJVCikrpb5ccRgQvdXRN__IJ64T894/view?usp=sharing) and place it in the appropriate location, following the directory structure below:

```
V2X-ATCT
├── V2X-ATCT/
│   ├── data/
│   │   │   ├── semantic
│   │   │   ├── v2x_dataset
├── core/
├── config/
├── ...
```

## **2. Installation**

Enter the directory containing the Dockerfile in the terminal, and run the following command to install the program.

```shell
docker build -t v2x-atct .
```

## **3. Launch**

Start the server using the following command, and enter `http://172.17.0.2:5000` in your browser. Microsoft Edge is recommended, as other browsers may have blocking issues.

```shell
docker run -p 8501:8501 v2x-atct
```

## **4. Visualization**

If you wish to visualize the scene, after completing all the steps above, enter the URL in your browser to access the frontend interface. Adjust the parameters in the **Scenario Generation** module and click the "Generate" button. The scene generation is expected to take 20–30 minutes. Once completed, you can find the corresponding results in the **Scenario Data** module. Each folder represents a generated scene. To visualize the scene, run the following command and replace `${scene_dir}` with the name of your generated scene folder, for example: `2025-06-19_20:31:35`.




```shell
xhost +local:root

docker run -it --rm \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --device=/dev/dri:/dev/dri \
    v2x-atct \
    python V2X-ATCT/opencood/visualization/vis_data_sequence.py --data ${scene_dir}
```