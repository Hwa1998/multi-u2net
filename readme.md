# 多类别的 u2net 语义分割
## Reference
https://github.com/xuebinqin/U-2-Net

把 作者的 u2net 改成了 mutil u2net，能用就行，但没写验证阶段及 mIoU 的计算 =_=

## 1、改动如下

​	1、data_loader.py 中更改了对类别值的计算方式，使其在读掩码图时直接获得对应的类别值

​	2、更改了 loss 的计算方式，并且激活函数 从 sigmoid 更改为 softmax

## 2、预训练权重为 mutil_u2net.pth。其获取过程为

​	1、加载 作者开源的预训练权重 u2netp.pth

​	2、移除 u2netp.pth 权重的最后一层

​	3、保存新的预训练权重

​	4、预训练权重的路径为：saved_models/pretrain_model/mutil_u2net.pth

## 3、模型训练及推理

​	1、训练集的数据样式，参考多类别语义分割的 voc 格式

​		JPEGImages(三通道rgb图像) + SegmentationClass(单通道掩码图)

​	2、模型训练：train.py

​	3、模型推理：pytorch_infer.py

​	4、脚本的代码量很小，用自己的数据集训练模型的话跟着注释改对应的代码就行

## 4、权重文件的部署

​	1、torch2onnx.py

​		转 onnx 时调用的模型脚本是 u2net_onnx.py，训练模型时用的脚本是 u2net.py

​	2、opencv_infer.py、onnx_infer.py
