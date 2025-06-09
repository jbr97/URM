# Single Domain Generalization for Few-Shot Counting via Universal Representation Matching



## Setup

### Install the libraries

To run the code, install the following libraries: `Mindspore 2.5.0`, `mindcv`, `scipy`, `numpy` and `PIL`.

## Data Preparation
1.Download the FSC147 dataset as instructed in its [official repository](https://github.com/cvlab-stonybrook/LearningToCountEverything).

2.Download the FSCD_LVIS dataset as instructed in its [official repository](https://github.com/VinAIResearch/Counting-DETR).



### Generate prompt
    pyhton utils/genetate_prompt


### Generate density maps (optional)


    python utils/data.py --data_path <path_to_your_data_directory> --image_size 512 
    
where `<path_to_your_data_directory>` is the path to the dataset created in the previous step.


## Training

The training code is in the `train.py` script. `start_train.sh` and `start_train_zeroshot.sh` shows an example of how to run training on, make sure to modify the paths and data in `--data`, `--data_path`, `--val_data`, `--val_data_path`to point to your training/validation data and data path.



## Acknowledgements
We thank the following projects: [loca](https://github.com/djukicn/loca)
