# darknet53

This is an implementation of DarkNet53 network discussed in [yolov3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) by pytorch.
 
1、DarkNet53 classification  
---------------------------
darknet53，imagenet数据集上分布式训练，模型文件（darknet53.pth）下载   
前向测试脚本： inference_darknet53.py   
百度网盘链接：https://pan.baidu.com/s/1gRzKsec0xvVZENxbnPvJmw 提取码: 99bm    
谷歌网盘链接：https://drive.google.com/file/d/1VyTXsW3O29Vr-sX5VZCpQLy_3CV4EpYX/view?usp=sharing  

2、CspDarknet53 classificaton    
-----------------------------  
3、YOLOV4 object detection    
------------------------------  

imagenet数据集上分类accuracy:  
---------------------------  
| 模型        | top1 acc |  top5 acc |
| --------   | -----:   |   -----:  | 
| darknet53(分布式训练)        | 76.5220% |  93.102%  |  
| cspdarknet53     | to do |  to do  |

imagenet数据集上分类速度:  
------------------------  
| 模型  | cpu平均耗时(10次) | nvidia-p40平均耗时(10次) |
| ----- | ----------: | -----------: |
|darknet53(分布式训练)|  |  |
|cspdarknet53|     |      |

