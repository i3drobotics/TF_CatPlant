# TF Cat Plant Detection

## Install Dependencies
Install python 3.5+ *(tested up to Python v3.6.8)*

  * **Linux**
    ```
    sudo apt-get install python3.5
    ```
  * **Windows**

    Download and install python from [here](https://www.python.org/downloads/)

Install tensorflow pip package
```
python -m pip install tensorflow==1.15
```
Install opencv pip package
```
python -m pip install opencv-python
```
Install git

  * *Linux*
    ```
    sudo apt-get install git
    ```
  * **Windows**

    Download and install git from [here](https://git-scm.com/download/win)

Install git LFS

  * **Linux**
    ```
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install git-lfs
    git lfs install
    ```
  * **Windows**

    Download Windows Installer from [here](https://github.com/git-lfs/git-lfs/releases) then run:
    ```
    git lfs install
    ```
**[Optional]**

Install TensorFlow GPU

For faster training and classifying you can use the GPU implimentation of TensorFlow using CUDA. This requires a NVidia GPU with driver version >=410.48. See [here](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) for full compatibility details of CUDA. 
```
python -m pip install tensorflow-gpu==1.15
```
See [here](https://www.tensorflow.org/install/gpu) for detailed install instructions installing GPU tensorflow for Linux/Windows (Currently required to use v1.15).

## Clone repository
Clone this repository (with submodule)
```
git clone --recursive https://github.com/i3drobotics/TF_CatPlant.git
```

## Setup
See [here](https://github.com/i3drobotics/TF_Sub/blob/master/README.md) for details on training and classifying data with TF_Sub.

## Workspace
All machine learning scripts should be run in the directory [PATH TO REPO]/TF_Sub/models/research/object_detection. Set the current working directory to this directory in the terminal.
```
cd PATH_TO_REPO/TF_Sub/models/research/object_detection
```
*NOTE:* If you are running vscode tasks this is handled automatically.

## Update training
This repository contains a trained model using the images provided in 'images'. Should extra labelled images be added the following procedure will need to be followed to re-train the model.
After adding labelled data to the 'images' folder, run the following scripts one after another (***WARNING!*** this includes purging the previous model)
```
python purge.py dir=../../../..
python xml_to_csv.py --images_dir=../../../../images
python generate_tfrecord.py --csv_input=../../../../images/train_labels.csv --image_dir=../../../../images/train --output_path=../../../../train.record
python generate_tfrecord.py --csv_input=../../../../images/test_labels.csv --image_dir=../../../../images/test --output_path=../../../../test.record
[Linux]
python train.py --logtostderr --train_dir=../../../../training/ --pipeline_config_path=../../../../training/faster_rcnn_inception_v2_coco_linux.config
[Windows]
python train.py --logtostderr --train_dir=../../../../training/ --pipeline_config_path=../../../../training/faster_rcnn_inception_v2_coco_win.config
```
Or run the vscode task: **TF: Train: New** (***WARNING!*** this includes purging the previous model)

Whilst training this will create a checkpoint every 10 mins. A checkpoint allows for monitoring while the training process is running as well as a safety should a problem occur whilst training process is running. 

The training can be monitored using 'Tensorboard'. This creates a local server the shows the training loss graphs. Run the following command in the terminal to star the server:
```
tensorboard --logdir=../../../../training --host localhost --port 8088
```
This can then be viewed by going to the following address in your default browser http://localhost:8088/

There are two vscode tasks for this: 

**TF: tensorboard: start** to start the server. 

**TF: tensorboard: show** to open the address in google-chrome

![Screenshot of loss graphs](https://github.com/i3drobotics/TF_CatPlant/raw/master/screenshots/Loss_Graphs.PNG "Tensorboard Loss graphs")

Training will continue untill 200000 steps is reached as this is expected to be a point where no extra learning can be acheived. However, this can be extended if needed by editing faster_rcnn_inception_v2_coco_*OS*.config **line 113**. 

Training can be stopped when the graphs show a loss of less than 0.1. This can be done by simply pressing CTRL+C in the terminal that is running the training.

The same 'train.py' python script will resume a training session if there is checkpoint data in the 'training' folder.

## Update model
A tensorflow checkpoint file is not useful in its current state so an intererence graph must be created from the model. This can be done using the python script 'export_inference_graph.py':
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path=../../../../training/faster_rcnn_inception_v2_coco_OS.config --trained_checkpoint_prefix=../../../../training/model.ckpt-XXXX --output_directory=../../../../inference_graph
```
Where 'model.ckpt-*XXXX*' is the checkpoint file with the largest step count e.g. model.ckpt-107679.

And 'faster_rcnn_inception_v2_coco_*OS*.config' is the pipline config file found in 'training/'. This should be 'faster_rcnn_inception_v2_coco_win.config' if running on windows and 'faster_rcnn_inception_v2_coco_linux.config' if running on linux. This will overwrite the contents of pipline.config. 

Or running the vscode task: **TF: Export graph** then editing the prompt displaying 'model.ckpt-XXXX' to the checkpoint file with the largest step count e.g. model.ckpt-107679. The pipeline config file is chosen automatically.

## Classify
Now that the model has been trained and exported it can be tested using the 'unknown' dataset. This will run the unknown images through the inference graph and classify the cats/plants along with a bounding box. This can be run with the python script: 
```
python classify.py --test_image=../../../../images/unknown/unknownX.jpg --inference_graph=../../../../inference_graph --training_dir=../../../../training
```
Where 'unknown*X*.jpg' is the name of the unknow image to classify. 

Or running the vscode task: **TF: Classify** then editing the prompt displaying unknown*X* to the unknown image filename to classify e.g. 'unknown2'. 

![Screenshot of classified image](https://github.com/i3drobotics/TF_CatPlant/raw/master/screenshots/TestImage_classified.PNG "Classified image")

Also avaiable for testing the classification is a script that runs openCV Video Capture to classify based on the image from a usb camera or video file. This can be run with the following command:
```
python classify_vid.py --inference_graph=../../../../inference_graph --training_dir=../../../../training [--vid=PATH_TO_VIDEO / --cam=0]
```
For a video file use the option '--vid'.

For a usb camera use the option '--cam'.

### Standalone
For standlone running of the classification there is a 'standalone' folder which holds all the scripts needed to load a model and run the classification. Once the model in the repository has been trained and exported, copy the 'frozen_inference_graph.pb' from 'inference_graph' folder in the repository to the 'Model' folder in the standalone directory. Also copy the 'labelmap.pbtxt' inside 'training' to the same 'Model' directory in the standalone folder. Alternatively run the following script:
```
cd PATH_TO_REPO/TF_Sub
generate_standalone.py --inference_graph=../inference_graph --training_dir=../training --standalone_dir=../standalone
```
Or copy the required files into the standalone directory by running the vscode task: **TF: Generate standalone**

This 'standalone' folder could now be copied anywhere required and it has all the data and scripts needed to classify. 
Classification can be run the same way as previously as this same script is included in the 'standalone' directory:
```
cd PATH_TO_REPO/standalone/scripts/object_detection
classify_vid.py --inference_graph=../../Model --training_dir=../../Model --output_dir=../../Result --cam=0 --flip --split
```
Provided is a vscode task within the standalone directory. To run this open the standalone directory with vscode then hit F1-> Tasks: Run Task -> TF: Classify: Camera Feed