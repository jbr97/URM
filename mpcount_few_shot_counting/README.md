
## Requirements
* Python 3.10.12
* Mindspore 2.5.0
* mindcv 
* Others specified in [requirements.txt](requirements.txt)

## Data Preparation
1.Download the FSC147 dataset as instructed in its [official repository](https://github.com/cvlab-stonybrook/LearningToCountEverything).

2.Download the FSCD_LVIS dataset as instructed in its [official repository](https://github.com/VinAIResearch/Counting-DETR).

## Training
```
python main.py --task train --config configs/fsc147_lvis_train.yml
```
You may edit the `.yml` config file as you like.

## Acknowledgements
We thank the following projects: [MPCount](https://github.com/Shimmer93/MPCount),[loca](https://github.com/djukicn/loca)

## Citiation
If you find this repository useful for your research, please use the following:

```
x
```
