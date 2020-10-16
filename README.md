# darknet53

This is an implementation of DarkNet53 network discussed in [yolov3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) by pytorch.
 
1、DarkNet53 classification  
---------------------------
darknet53，imagenet数据集上分布式训练，模型文件（darknet53.pth）下载   
训练脚本： python main.py --dist-url env:// --dist-backend nccl --world-size 4 imagenet2012_path  
训练的时候使用了4张p40显卡，world-size设为4  
前向测试脚本： inference_darknet53.py   
百度网盘链接：https://pan.baidu.com/s/1gRzKsec0xvVZENxbnPvJmw 提取码: 99bm    
谷歌网盘链接：https://drive.google.com/file/d/1VyTXsW3O29Vr-sX5VZCpQLy_3CV4EpYX/view?usp=sharing  

2、CspDarknet53 classificaton    
-----------------------------    
cspdarknet53,imagenet数据集上分布式训练，模型文件（）下载  
训练脚本： python main.py --dist-url env:// --dist-backend nccl --world-size 6 imagenet2012_path  
训练的时候使用了6张p40显卡，world-size设为6  
前向测试脚本：  
百度网盘链接：  
谷歌网盘链接：  

3、YOLOV4 object detection    
------------------------------  
https://github.com/njustczr/yolov4-pytorch    

imagenet数据集上分类accuracy:  
---------------------------  
| 模型        |  input_size  | top1 acc |  top5 acc |
| --------   | -----: |  -----:   |   -----:  | 
| darknet53(分布式训练)        | 256x256 | 76.522% |  93.102%  |  
| cspdarknet53     |  256x256  | to do |  to do  |

|  模型 |input_size|warmup|polynomial-lr|step-lr|mish|label-smoothing|cut-mix|epoch|top1 acc|top5 acc|
| -----| ---------: |-------:|---------------:|-------:|-----:|----------------:|--------:|-----:|----------:|---------:|
| cspdarknet53|     |        |                |        |      |                 |         |      |           |          | 


imagenet数据集上分类速度:  
------------------------  
| 模型  | input_size | intel i5 cpu平均耗时(10次) | nvidia-p40平均耗时(10次) |
| ----- | -------: | ----------: | -----------: |
|darknet53(分布式训练)| 256x256 | 0.2s  | 0.017s |
|cspdarknet53| 256x256 |  to do   |  to do  |

