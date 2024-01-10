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
Please register and download [Inria Aerial](https://project.inria.fr/aerialimagelabeling/) dataset and [DeepGlobe](https://competitions.codalab.org/competitions/18468) dataset
We follow [FCtL](https://github.com/liqiokkk/FCtL) to split two datasets.

## Citation
If you use this code or our results for your research, please cite our paper.
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
```
@inproceedings{li2021contexts,
  title={From Contexts to Locality: Ultra-high Resolution Image Segmentation via Locality-aware Contextual Correlation},
  author={Li, Qi and Yang, Weixiang and Liu, Wenxi and Yu, Yuanlong and He, Shengfeng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={7252--7261},
  year={2021}
}
```
