# Memory-Constrained Semantic Segmentation for Ultra-High Resolution UAV Imagery
[Memory-Constrained Semantic Segmentation for Ultra-High Resolution UAV Imagery](https://arxiv.org/abs/2310.04721)  
Qi Li, Jiaxin Cai, Jiexin Luo, Yuanlong Yu, Jason Gu, Jia Pan, Wenxi Liu  
Accepted to [IEEE RA-L 2024](https://ieeexplore.ieee.org/document/10380673)
## Abstract
Ultra-high resolution image segmentation poses a formidable challenge for UAVs with limited computation resources. Moreover, with multiple deployed tasks (e.g., mapping, localization, and decision making), the demand for a memory efficient model becomes more urgent. This paper delves into the intricate problem of achieving efficient and effective segmentation of ultra-high resolution UAV imagery, while operating under stringent GPU memory limitation. To address this problem, we propose a GPU memory-efficient and effective framework. Specifically, we introduce a novel and efficient spatial-guided high-resolution query module, which enables our model to effectively infer pixel-wise segmentation results by querying nearest latent embeddings from low-resolution features. Additionally, we present a memory-based interaction scheme with linear complexity to rectify semantic bias beneath the high-resolution spatial guidance via associating cross-image contextual semantics. For evaluation, we perform comprehensive experiments over public benchmarks under both conditions of small and large GPU memory usage limitations. Notably, our model gains around 3% advantage against SOTA in mIoU using comparable memory. Furthermore, we show that our model can be deployed on the embedded platform with less than 8G memory like Jetson TX2.
![framework](https://github.com/liqiokkk/SGHRQ/blob/main/framework.png)
## Test and train
python==3.7, pytorch==1.10.0, and mmcv==1.7.0
### dataset
Please register and download [Inria Aerial](https://project.inria.fr/aerialimagelabeling/) dataset and [DeepGlobe](https://competitions.codalab.org/competitions/18468) dataset.  
We follow [FCtL](https://github.com/liqiokkk/FCtL) to split two datasets.  
Create folder named 'InriaAerial', its structure is 
```
    InriaAerial/
    ├── imgs
       ├── train
          ├── xxx_sat.tif
          ├── ...
       ├── test
       ├── val
    ├── labels
       ├── train
          ├── xxx_mask.png(two values:0-1)
          ├── ...
       ├── test
       ├── val
```
Create folder named 'DeepGlobe', its structure is
```
    DeepGlobe/
    ├── img_dir
       ├── train
          ├── xxx_sat.jpg
          ├── ...
       ├── val
       ├── test
    ├── rgb2id
      ├── train
          ├── xxx_mask.png(0-6)
          ├── ...
      ├── val
      ├── test
```
### test
Please download our pretrianed-model [here](https://drive.google.com/drive/folders/178LTNGOE-4zc5LN8rFVF2DihOjwsXBS7?usp=drive_link)
```
python ./test.py configs/SGHRQ/SGHRQ_r18-d8_2000x2000_40k_InriaAerial.py InriaAerial.pth --eval mIoU  
python ./test.py configs/SGHRQ/SGHRQ_r18-d8_1224x1224_80k_DeepGlobe.py DeepGlobe.pth --eval mIoU
```
### train
Please download STDC pretrianed-model [here](https://github.com/MichaelFan01/STDC-Seg)
```
python ./train.py configs/SGHRQ/SGHRQ_r18-d8_2000x2000_40k_InriaAerial.py  
python ./train.py configs/SGHRQ/SGHRQ_r18-d8_1224x1224_80k_DeepGlobe.py
```
## Citation
If you use this code or our results for your research, please cite our papers.
```
@ARTICLE{10380673,
  author={Li, Qi and Cai, Jiaxin and Luo, Jiexin and Yu, Yuanlong and Gu, Jason and Pan, Jia and Liu, Wenxi},
  journal={IEEE Robotics and Automation Letters}, 
  title={Memory-Constrained Semantic Segmentation for Ultra-High Resolution UAV Imagery}, 
  year={2024},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/LRA.2024.3349812}}
```
