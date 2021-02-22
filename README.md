# Towards More Explainability: Concept Knowledge Mining Network for Event Recognition

This is a PyTorch implementation of the model described in our paper:

> Z. Qi, S. Wang, C. Su, L. Su, Q. Huang, and Q. Tian. Towards More Explainability: Concept Knowledge Mining Network for Event Recognition. ACM MM 2020.

# Dependencies

  * Pytorch 1.0.1
  * Cuda 9.0.176
  * Cudnn 7.4.2
  * Python 3.6.8 

# Data

## Dataset Prepare

1. Download the pre-trained concept detector weights form [Google Grive](https://drive.google.com/drive/folders/1jPssDmtePpwiJEIdZtNqM6PlLeo5uSyo?usp=sharing) and put them in folder weights/

2. Download the FCVID dataset from http://bigvid.fudan.edu.cn/FCVID/.

3. The annotation information of each  dataset is provided in folder data/FCVID/video_labels.

4. Extract the video frames for each video and put the extracted frames in folder data/FCVID/frames/.

   For ActivityNet dataset ( http://activity-net.org/. ) , we use the latest released version of the dataset (v1.3).

# Train

* python main.py --gpu_ids 0,1 --dataset FCVID  --no_test

  for other hyperparameters, please refer to opts.py file.

# Test

* Pretrained model weigths are avaiable in [Google Grive](https://drive.google.com/drive/folders/1FBNcJ4lPAviR68eU6ukYUFCWOocBlNhO?usp=sharing) 
* Download the pre-trained weights and put them in folder results/

* python main.py --gpu_ids 0,1 --dataset FCVID  --resume_path pretrained_model/CKMN.pth --no_train --test_crop_number 1


# **Citation**

Please cite our paper if you use this code in your own work:

```
@inproceedings{qi2020towards,
  title={Towards More Explainability: Concept Knowledge Mining Network for Event Recognition},
  author={Qi, Zhaobo and Wang, Shuhui and Su, Chi and Su, Li and Huang, Qingming and Tian, Qi},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={3857--3865},
  year={2020}
}
```

# Contcat

If you have any problem about our code, feel free to contact

- [zhaobo.qi@vipl.ict.ac.cn](mailto:zhaobo.qi@vipl.ict.ac.cn)
