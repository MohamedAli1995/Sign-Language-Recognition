# Sign-Language-Recognition
Sign recognition network that predicts a digit giving an image of the sign.<br>

This project structure follows the **best practice tensorflow folder structure of** [Tensorflow Best Practice](https://github.com/MrGemy95/Tensorflow-Project-Template) 

# Table of contents
- [Project structure](#project-structure)
- [Download pretrained models](#Download-pretrained-models)
- [Dependencies](#install-dependencies)
- [Config file](#config-file)
- [How to train](#How-to-Train)
- [How to predict](#Make-predictions-with-pretrained-models)
- [Implementation details](#Implementation-details)
     - [Preprocessing](#Gesture-recognition-model-preprocessing)
          - [Dataset Shuffling][#Shuffling-dataset]
          - [Splitting train, val and test][#Splitting-dataset]
          - [Change input dimensions][#Changing-input-dimensions]
          - [Normalizing batches to zero mean][#Normalizing-batches]
          - [Data Augmentation][#Data-augmentation]
          
     - [Sign Recognition model architecture](#Gesture-recognition-model-arch)
     - [Model training](#Model-training)


# Project structure
--------------

```
├── Configs
│   └── config_model.json  - Contains the paths used and config of the models(learning_rate, num_epochs, ...)
│ 
├──  base
│   ├── base_model.py   - This file contains the abstract class of all models used.
│   ├── base_train.py   - This file contains the abstract class of the trainer of all models used.
│   └── base_test.py    - This file contains the abstract class of the testers of all models used.
│
├── models              - This folder contains 1 model for sign language detection.
│   └── gesture_recognition_model.py  - Contains the architecture of the gesture/sign recognition model used
│
│
├── trainer             - This folder contains trainers used which inherit from BaseTrain.
│   └── gesture_recognition_trainer.py - Contains the trainer class of the gesture recognition model.
│ 
|
├── testers             - This folder contains testers used which inherit from BaseTest.
│   └── sentiment_tester.py - Contains the tester class of the gesture recognition model.
│ 
| 
├──  mains 
│    └── main.py  - responsible for the whole pipeline.
|
│ 
├──  data _loader 
│    ├── data_generator.py  - Contains DataGenerator class which handles Sign Language dataset.
│    └── preprocessing.py   - Contains helper functions for preprocessing Sign Language dataset.
| 
└── utils
     ├── config.py  - Contains utility functions to handle json config file.
     ├── logger.py  - Contains Logger class which handles tensorboard.
     └── utils.py   - Contains utility functions to parse arguments and handle pickle data. 
```


# Download pretrained models:
Pretrained models can be found at saved_models/checkpoint

# Install dependencies

* Python3.x <br>

* [Tensorflow](https://www.tensorflow.org/install)

* Tensorboard[optional] <br>

* Numpy
```
pip3 install numpy
```

* scipy version 1.1.0
```
pip3 install scipy==1.1.0
```

* Bunch
```
pip3 install bunch
```

* Pandas
```
pip3 install pandas
```

* tqdm
```
pip3 install tqdm
```

# Config File
In order to train, pretrain or test the model you need first to edit the config file:
```
{
  "num_epochs": 200,               - Numer of epochs to train the model if it is in train mode.
  "learning_rate": 0.001,         - Learning rate used for training the model.
  "state_size": [64, 64, 1],      - Holding the input size ( our state).
  "batch_size": 256,               - Batch size for training, validation and testing sets(#TODO: edit single batch_size per mode)
  "val_per_epoch": 1,              - Get validation set acc and loss per val_per_epoch. (Can be ignored).
  "max_to_keep":1,                 - Maximum number of checkpoints to keep.
  "per_process_gpu_memory_fraction":1, - Usage percentage of the GPU for trainin, I needed it...

  "train_data_path":"path_to_training_set",                      - Path to training data.
  "test_data_path":"path_to_test_set",                           - Path to test data.
  "checkpoint_dir":"path_to_store_the_model_checkpoints",        - Path to checkpoints store location/ or loading model.
  "summary_dir":"path_to_store_model_summaries_for_tensorboard",  - Path to summaries store location/.

}
```

# How to Train
In order to train, pretrain or test the model you need first to edit the config file that is described at [config file](#config-file).<br>
To train a Gesture Recognition model:<br>
set:<br>
```
"num_epochs":200,
"learning_rate":0.0001,
"batch_size":256,
"state_size": [64, 64, 1],

"train_data_path": set it to path of the training data e.g: "/content/train"
"checkpoint_dir": path to store checkpoints, e.g: "/content/saved_models/tiny_vgg_model/checkpoint/"
"summary_dir": path to store the model summaries for tensorboard, e.g: "/content/saved_models/tiny_vgg_model/summary/"
```
Then change directory to the project's folder and run:
```
python3.6 -m src.mains.main --config path_to_config_file
```
# Make predictions with pretrained models
### To make predictions using single image as input input:<br>
Configure the config file to the path of the model checkpoint.<br>
cd to project folder.<br>
```
python3.6 -m src.mains.main --config "path_to_config_file" -i "path_to_image"
```

### To make predictions using path for test images input:<br>
Configure the config file to the path of the model checkpoint.<br>
cd to project folder.<br>
```
python3.6 -m src.mains.main --config "path_to_config_file" -t "path_to_images_folder"
```


# Implementation details
## Gesture recognition model preprocessing
talk about preprocessing
### Shuffling dataset
In order to decrease the probability of overfitting, training set is shuffled every new epoch<br>
```
indices_list = [i for i in range(self.x_train.shape[0])]  # Training examples.
shuffle(indices_list)
```

### Splitting dataset
Dataset is splitted into 3 segments, training set(80%), validation set(10%) and testing set (10%)<br>

### Changing input dimensions
Input image is a 100x100x3 images, they are downsampled to a 64x64x1 gray-scale image, as color information are redundant and meaningless in this task (we don't want to differentiate between white and dark hands..) <br>
```
img = scipy.misc.imresize(img, (64, 64))
```

### Normalizing batches
In order to change the input images values to a common scale, data is normalized to be a zero-mean <br>
'''
img = (img - img.min()) / (img_range + 1e-5)  #1e-5 is to prevent division by zero
'''

### Data augmentation
TODO: Augment training set.
Data augmentation will help in increasing the model performance <br>

## Gesture recognition model arch
<img src="https://github.com/MohamedAli1995/Sign-Language-Recognition/blob/master/diagrams/model_diagram.png"
     alt="Image not loaded" style="float: left; margin-right: 10px;" />

## Model Training
 I trained the Sign Language model by splitting training_data into train/val/test with ratios 8:1:1 for 300 epochs<br>
 Acheived val accuracy of 97.5781261920929%, val_loss of 0.06723332<br>
 training accuracy of 99%, training_loss of 0.26895<br><br>

model val_acc <br>
<img src="https://github.com/MohamedAli1995/Sign-Language-Recognition/blob/master/saved_models/diagrams/val_acc.png" alt="Image not loaded" style="float: left; margin-right: 10px;" />

     alt="Image not loaded"
     style="float: left; margin-right: 10px;" />

and val_loss <br>
<img src="https://github.com/MohamedAli1995/Sign-Language-Recognition/blob/master/saved_models/diagrams/val_loss.png"
     alt="Image not loaded"
     style="float: left; margin-right: 10px;" />
     
## model testing
   Acheived testing accuracy of 97.38948345184326% on 10% of the dataset (unseen in training process).<br>
   with test_loss:  0.049882233.
   
