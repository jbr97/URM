
## Requirements
* Python 3.10.12
* Others specified in [requirements.txt](requirements.txt)

## Data Preparation
1.Download the FSC147 dataset as instructed in its [official repository](https://github.com/cvlab-stonybrook/LearningToCountEverything).

2.Download the FSCD_LVIS dataset as instructed in its [official repository](https://github.com/VinAIResearch/Counting-DETR).

## Training
```
python main.py --task train --config configs/fsc147_lvis_train.yml
```
You can edit the `.yml` config file as you like.

## Acknowledgements
We thank the following projects: [MPCount](https://github.com/Shimmer93/MPCount),[LOCA](https://github.com/djukicn/loca)

## Citiation
If you find this repository useful for your research, please use the following:

```
@inproceedings{chen2025single,
  title={Single Domain Generalization for Few-Shot Counting via Universal Representation Matching},
  author={Chen, Xianing and Huo, Si and Jiang, Borui and Hu, Hailin and Chen, Xinghao},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={4639--4649},
  year={2025}
}
@inproceedings{peng2024single,
  title={Single domain generalization for crowd counting},
  author={Peng, Zhuoxuan and Chan, S-H Gary},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={28025--28034},
  year={2024}
}
```
