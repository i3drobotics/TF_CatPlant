# TF Object Detection Template

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

## Create repository
Clone this repository (with submodule)
```
git clone --recursive https://github.com/i3drobotics/ML_TF_Object_Detect_Template.git
```
Create a new git repository
- Go to [GitHub](www.github.com).
- Log in to your account.
- Click the new repository button in the top-right. You’ll have an option there to initialize the repository with a README file, but don’t.
- Click the “Create repository” button.
- Make a note of the repository url

Set remote to your git repository url
```
cd ML_TF_Object_Detect_Template
git remote set-url https://github.com/USERNAME/REPOSITORY.git
```
Also you can rename the folder to something that makes sense for your dataset (e.g 'ML_TF_CatPlants_Detect').

## VSCode
VS Code tasks are provided in this repository to make running quick and easy. 

Install VSCode [here](https://code.visualstudio.com/Download)

Open the repository in VSCode by running vscode and then: file -> open folder -> [PATH TO REPO]

*Note:* There appears to be an issue with VSCode where it doesn't load environment variables in Windows unless run as an admin. This can cause problems with git LFS so start vscode as an admin: start -> vscode -> right-click -> run as admin

To run these tasks open this repository inside VS code and press F1. Enter 'Tasks' and look for 'Tasks: Run task'. This will then show a list of tasks that can be run. 

To make sure the 'python.pythonPath' used in the tasks is set open a python script within VSCode and select the opropriate python interpretor. This should create settings.json in .vscode and set the 'python.pythonPath' variable.

## Workspace
All machine learning scripts should be run in the directory models/research/object_detection. Set the current working directory to this directory in the terminal.
```
cd PATH_TO_REPO/model/research/object_detection
```
*NOTE:* If you are running vscode tasks this is handled automatically.

## Demo
### Dataset
For machine learning to work a labelled dataset is needed to train the model. This repository demonstrates training a model to detect cats and plants. The labelled images are provided in 'images'.

There are sorted into three categories; 'Train', 'Test', and 'Unknown'. 'Train' images are the images used to train the machine learning. 'Test' are used in the training process as a measure for how well the training is performing. 'Unknown' are used as images never seen by the training process to manually test the model after training. 

All of the images in 'Train' and 'Test' images are labelled with what and where the cats and plants are in the images using bounding boxes. These labels take the form of xml documents for each images. These were created using the labelimg program provided in this repository. Details on using this program can be found in the programs repository [here](https://github.com/tzutalin/labelImg).

For use in Tensorflow these xml files need to be converted to a single CSV file the lists the filenames and bounding boxes for each dataset. This is performed with the following python script: 
```
python xml_to_csv.py
```
Or using the vscode task: **TF: XML to CSV**

Tensorflow cannot actually read csv files directly so TFRecord files must be generated for each dataset. This is performed with the 'generate_tfrecord.py' python script:
```
cd PATH_TO_REPO/model/research/object_detection
python generate_tfrecord.py --csv_input=../../../../images/train_labels.csv --image_dir=../../../../images/train --output_path=../../../../train.record
python generate_tfrecord.py --csv_input=../../../../images/test_labels.csv --image_dir=../../../../images/test --output_path=../../../../test.record
```
Or using the vscode tasks: **TF: generateTFrecords: train**, **TF: generateTFrecords: test**

### Train
Training is performed with the 'train.py' python script: 
```
python train.py --logtostderr --train_dir=../../../../training/ --pipeline_config_path=../../../../training/faster_rcnn_inception_v2_coco_OS.config
```
Where 'faster_rcnn_inception_v2_coco_*OS*.config' is the pipline config file found in 'training/'. This should be 'faster_rcnn_inception_v2_coco_win.config' if running on windows and 'faster_rcnn_inception_v2_coco_linux.config' if running on linux. This will overwrite the contents of pipline.config. 

This will create a checkpoint every 10 mins. A checkpoint allows for monitoring while the training process is running as well as a safety should a problem occur whilst training process is running. 

The training can be monitored using 'Tensorboard'. This creates a local server the shows the training loss graphs. Run the following command in the terminal to star the server:
```
cd PATH_TO_REPO
tensorboard --logdir=../../../../training --host localhost --port 8088
```
This can then be viewed by going to the following address in your default browser http://localhost:8088/

There are two vscode tasks for this: 

**TF: tensorboard: start** to start the server. 

**TF: tensorboard: show** to open the address in google-chrome *(linux only)*

Training will continue untill 200000 steps is reached as this is expected to be a point where no extra learning can be acheived. However, this can be extended if needed by editing faster_rcnn_inception_v2_coco_*OS*.config **line 113**. 

Training can be stopped when the graphs show a loss of less than 0.1. This can be done by simply pressing CTRL+C in the terminal that is running the training.

The same 'train.py' python script will resume a training session if there is checkpoint data in the 'training' folder.

## Export graph
A tensorflow checkpoint file is not useful in its current state so an intererence graph must be created from the model. This can be done using the python script 'export_inference_graph.py':
```
cd PATH_TO_REPO/model/research/object_detection
python export_inference_graph.py --input_type image_tensor --pipeline_config_path=../../../../training/faster_rcnn_inception_v2_coco_OS.config --trained_checkpoint_prefix=../../../../training/model.ckpt-XXXX --output_directory=../../../../inference_graph
```
Where 'model.ckpt-*XXXX*' is the checkpoint file with the largest step count e.g. model.ckpt-107679.

And 'faster_rcnn_inception_v2_coco_*OS*.config' is the pipline config file found in 'training/'. This should be 'faster_rcnn_inception_v2_coco_win.config' if running on windows and 'faster_rcnn_inception_v2_coco_linux.config' if running on linux. This will overwrite the contents of pipline.config. 

Or running the vscode task: **TF: Export graph** then editing the prompt displaying 'model.ckpt-XXXX' to the checkpoint file with the largest step count e.g. model.ckpt-107679. The pipeline config file is chosen automatically.

### Classify
Now that the model has been trained and exported it can be tested using the 'unknown' dataset. This will run the unknown images through the inference graph and classify the cats/plants along with a bounding box. This can be run with the python script: 
```
cd PATH_TO_REPO/model/research/object_detection
python classify.py --test_image=../../../../images/unknown/${input:unknown_image_file}.jpg --inference_graph=../../../../inference_graph --training_dir=../../../../training
```
Where 'unknown*X*.jpg' is the name of the unknow image to classify. 

Or running the vscode task: **TF: Classify** then editing the prompt displaying unknown*X* to the unknown image filename to classify e.g. 'unknown2'. 

## Custom dataset
Once familiar with running the demo dataset you may want to edit the dataset and classes for your own requirements. The following section details the steps needed to change the workspace from plants and cats to plants, cats and dogs. 

### Data
In order to add a new label classifier some images are needed of the new class. In this case we are adding dogs so 20+ images of dogs should be added to the images in 'images/train' and 5+ images of dogs should be added in 'images/test'. To test completely unseen images once training is complete some more images should be added to 'images/unknown' that are later used to test model. The test dataset is used to create the loss metric while training so images in this folder will match very well. The unknown dataset is used to test the model on completely unseen data that the model has no idea about to give a better idea as the effectiveness of generality. 

### Labeling
The new images in test and train need to be labelled so they can be used in the training process. Included in this repo is a program called 'labelImg' that can be used to tag images with bounding boxes to label the location and class of objects in an image.
  * **Linux**
  ```
  cd [PATH TO REPO]/programs/labelimg
  sudo ./labelImg
  ```
  * **Windows**
  ```
  [PATH TO REPO]/programs/labelImg/labelImg.exe
  ```
  
After labeling the classes in an images hit 'Save' to create an XML file for that image. This will appear in the same folder as the image.

### Update parameters
As we have added a class to the dataset the config files need to be updated to inform the scripts of the new class. 

The 'labelmap.pbtxt' in 'training' needs to be edited to include the new class:
```
item {
  id: 1
  name: 'plant'
}

item {
  id: 2
  name: 'cat'
}

# add this to end of file
item {
  id: 3
  name: 'dog'
}
```

The files 'faster_rcnn_inception_v2_coco_win.config' and 'faster_rcnn_inception_v2_coco_linux.config' in 'training' need to be edited to update the number of classes. (**line 10**)
```
...

9:  faster_rcnn {
10:     num_classes: 3 #edit this line
11:     image_resizer {

...
```

### Purge
As we are creating a new dataset any residule training models from the demo should be deleted. This can be done by running:
```
cd PATH_TO_REPO/model/research/object_detection
python purge.py dir=../../../..
```

Or running the vscode task: **TF: Purge**

### Run
Now the workspace is ready you can run all the steps from the previous section **'Demo'**. Run the following scripts one after another (this includes purging the demo model):
```
cd PATH_TO_REPO/model/research/object_detection
python purge.py dir=../../../..
python xml_to_csv.py --images_dir=../../../../images
python generate_tfrecord.py --csv_input=./../../../images/train_labels.csv --image_dir=./../../../images/train --output_path=./../../../train.record
python generate_tfrecord.py --csv_input=./../../../images/test_labels.csv --image_dir=./../../../images/test --output_path=./../../../test.record
python train.py --logtostderr --train_dir=./../../../training/ --pipeline_config_path=./../../../training/faster_rcnn_inception_v2_coco_OS.config
```
Or run the vscode task: **TF: Train: New** (this includes purging the demo model)