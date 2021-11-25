# ContourletNet: A Generalized Rain Removal Architecture Using Multi-Direction Hierarchical Representation<br> (Accepted by BMVC'21)


![image](/img_repo/example.png)


# Abstract:
Images acquired from rainy scenes usually suffer from bad visibility which may damage the performance of computer vision applications. The rainy scenarios can be categorized into two classes: moderate rain and heavy rain scenes. Moderate rain scene mainly consists of rain streaks while heavy rain scene contains both rain streaks and the veiling effect (similar to haze). Although existing methods have achieved excellent performance on these two cases individually, it still lacks a general architecture to address both heavy rain and moderate rain scenarios effectively. In this paper, we construct a hierarchical multi-direction representation network by using the contourlet transform (CT) to address both moderate rain and heavy rain scenarios. The CT divides the image into the multi-direction subbands (MS) and the semantic subband (SS). First, the rain streak information is retrieved to the MS based on the multi-orientation property of the CT. Second, a hierarchical architecture is proposed to reconstruct the background information including damaged semantic information and the veiling effect in the SS. Last, the multi-level subband discriminator with the feedback error map is proposed. By this module, all subbands can be well optimized. This is the first architecture that can address both of the two scenarios effectively.


[[Paper Download]](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_ALL_Snow_Removed_Single_Image_Desnowing_Algorithm_Using_Hierarchical_Dual-Tree_ICCV_2021_paper.pdf)
[[Supplementary Material Download]](https://ntucc365-my.sharepoint.com/:b:/g/personal/f05943089_ntu_edu_tw/EXjU8U85nMZMkoHwqVCO_QEBlWvz9U803iinqfkLv3QrZg?e=3k0diD)
[[Slide Download]](https://ntucc365-my.sharepoint.com/:b:/g/personal/f05943089_ntu_edu_tw/EVUaKr-l1UNDoUeuInao0RkB6kv5MDMfUcUCNp96rRZeTA?e=5LYZSC)

You can also refer our previous works on other low-level vision applications!

Desnowing-[[HDCWNet]](https://github.com/weitingchen83/ICCV2021-Single-Image-Desnowing-HDCWNet) (ICCV'21) and [[JSTASR]](https://github.com/weitingchen83/JSTASR-DesnowNet-ECCV-2020)(ECCV'20)<br>
Dehazing-[[PMS-Net]](https://github.com/weitingchen83/PMS-Net)(CVPR'19) and [[PMHLD]](https://github.com/weitingchen83/Dehazing-PMHLD-Patch-Map-Based-Hybrid-Learning-DehazeNet-for-Single-Image-Haze-Removal-TIP-2020)(TIP'20)<br>
Image Relighting-[[MB-Net]](https://github.com/weitingchen83/NTIRE2021-Depth-Guided-Image-Relighting-MBNet) (NTIRE'21 1st solution) and [[S3Net]](https://github.com/dectrfov/NTIRE-2021-Depth-Guided-Image-Any-to-Any-relighting) (NTIRE'21 3 rd solution)<br>


# Network Architecture

![image](/img_repo/architecture.png)



# Experimental Results
## Quantitative Evaluation
![image](/img_repo/quantitative_heavy.png)
![image](/img_repo/quantitative_moderate.png)

## Qualitative Evaluation
![image](/img_repo/qualitative_heavy.png)
![image](/img_repo/qualitative_moderate.png)

# Setup and environment

To generate the recovered result you need:

1. Python 3
2. CPU or NVIDIA GPU + CUDA CuDNN
3. tensorflow 1.15.0
4. keras 2.3.0
5. dtcwt 0.12.0

Training
```
python ./train.py --logPath ./your_log_path --dataPath /path_to_data/data.npy --gtPath /path_to_gt/gt.npy --batchsize batchsize --epochs epochs --modelPath ./path_to_exist_model/model_to_load.h5 --validation_num number_of_validation_image --steps_per_epoch steps_per_epoch
```

*data.npy should be numpy of training image whose shape is (number_of_image, 480, 640, 3). The range is (0, 255) and the datatype is uint8 or int.<br>
*gt.npy should be numpy of ground truth image, whose shape is (number_of_image, 480, 640, 3). The range is (0, 255) and datatype is uint8 or int.

Example:
```
python ./train.py --logPath ./log --dataPath ./training_data.npy --gtPath ./training_gt.npy --batchsize 3 --epochs 1500 --modelPath ./previous_log/preivious_model.h5 --validation_num 200 --steps_per_epoch 80
```



Testing
```
$ python ./predict.py -dataroot ./your_dataroot -datatype datatype -predictpath ./output_path -batch_size batchsize
```
*datatype default: tif, jpg ,png

Examples
```
$ 
python ./predict.py -dataroot ./testImg -predictpath ./p -batch_size 3
python ./predict.py -dataroot ./testImg -datatype tif -predictpath ./p -batch_size 3
```


The pre-trained model can be downloaded from: https://ntucc365-my.sharepoint.com/:u:/g/personal/f05943089_ntu_edu_tw/EZtus9ex-GtNukLuSxWGmPIBEJIzRFMbEl0dFeZ_oTQnVQ?e=xnfqFL. 
Put the "finalmodel.h5" to the 'modelParam'.


# Citations
Please cite this paper in your publications if it is helpful for your tasks:    

Bibtex:
```
@inproceedings{chen2021contour,
  title={ContourletNet: A Generalized Rain Removal Architecture Using Multi-Direction Hierarchical Representation},
  author={Chen, Wei-Ting and Tsai, Cheng-Che and Fang, Hao-Yu and and Chen, I-Hsiang and Ding, Jian-Jiun and Kuo, Sy-Yen},
  booktitle={Proceedings of the British Machine Vision Conference},
  year={2021}
}
```
