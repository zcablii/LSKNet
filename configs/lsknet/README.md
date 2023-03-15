# Oriented R-CNN

> [Large Selective Kernel Network for Remote Sensing Object Detection](https://arxiv.org/pdf/0000000.pdf)

<!-- [ALGORITHM] -->

## Abstract


Recent research on remote sensing object detection has largely focused on improving the representation of oriented bounding boxes but has overlooked the unique prior knowledge presented in remote sensing scenarios. Such prior knowledge can be useful because tiny remote sensing objects may be mistakenly detected without referencing a sufficiently long-range context, and the long-range context required by different types of objects can vary. In this paper, we take these priors into account and propose the Large Selective Kernel Network (LSKNet). LSKNet can dynamically adjust its large spatial receptive field to better model the ranging context of various objects in remote sensing scenarios. To the best of our knowledge, this is the first time that large and selective kernel mechanisms have been explored in the field of remote sensing object detection. Without bells and whistles, LSKNet sets new state-of-the-art scores on standard benchmarks, i.e., HRSC2016 (98.46% mAP), DOTA-v1.0 (81.64% mAP) and FAIR1M-v1.0 (47.87% mAP). Based on a similar technique, we rank 2nd place in 2022 the Greater Bay Area International Algorithm Competition

## Results and models
Imagenet 300-epoch pre-trained LSKNet-T backbone: https://pan.baidu.com/s/1CQHQBLR9UVW-7cxg7LB2Dg?pwd=84jd 

Imagenet 300-epoch pre-trained LSKNet-S backbone: https://pan.baidu.com/s/1o-Pq8k_7qcTgfTWiBve6uA?pwd=dm3w

DOTA1.0

|         Backbone         |  mAP  | Angle | lr schd | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| LSKNet_T (1024,1024,200) | 81.37 | le90  |   1x    |   2*8    | [lsk_t_fpn_1x_dota_le90](./lsk_t_fpn_1x_dota_le90.py) | [model](https://pan.baidu.com/s/1LtHb7ahPAcGJGPGfNo9EuQ?pwd=auu6) \| [log](https://pan.baidu.com/s/1PpC-Qu0dcDQ-iYM0kz5i5g?pwd=0sre) |
| LSKNet_S (1024,1024,200) | 81.64 | le90  |   1x    |    1*8     |            [lsk_s_fpn_1x_dota_le90](./lsk_s_fpn_1x_dota_le90.py.py)             |         [model](https://pan.baidu.com/s/1dYfSldDDWWlqRfljLlYhvA?pwd=v55f) \| [log](https://pan.baidu.com/s/1r6n5SZjEQvo5F1W2-ngkYQ?pwd=chxz)         |

FAIR1M-1.0

|         Backbone         |  mAP  | Angle | lr schd | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| LSKNet_S (1024,1024,200) | 47.87 | le90  |   1x    |    1*8     |            [lsk_s_fpn_1x_dota_le90](./lsk_s_fpn_1x_dota_le90.py)             |         [model](https://pan.baidu.com/s/1sXyi23PhVwpuMRRdwsIJlQ?pwd=izs8) \| [log](https://pan.baidu.com/s/1idHq3--oyaWK3GWYqd8brQ?pwd=zznm)         |

HRSC2016 

|         Backbone         |  mAP(07)  |   mAP(12)   |  Angle | lr schd | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download                                                                                                                                                                              |
| :----------------------: | :---: | :---: | :-----: | :-----: | :------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| LSKNet_S (1024,1024,200) | 90.65  |  98.46  | le90  |   3x    |    1*8     |            [lsk_s_fpn_3x_hrsc_le90](./lsk_s_fpn_3x_hrsc_le90.py)             |         [model](https://pan.baidu.com/s/1jRLb5m6tGab6BF1ky1JP6A?pwd=bosr) \| [log](https://pan.baidu.com/s/1f0i5oGn3QseKQLAGMcy4_w?pwd=kn0x)         |



## Citation

```bibtex
@article{li2023large,
  title   = {Large Selective Kernel Network for Remote Sensing Object Detection},
  author  = {Li, Yuxuan and Hou, Qibin and Zheng, Zhaohui and Cheng, Minging and Yang, Jian and Li, Xiang},
  journal={ArXiv},
  year={2023}
}
```
