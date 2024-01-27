import numpy as np

import cv2 as cv
from PIL import Image
import os
from skimage import transform
import torch
import time


def ToTensorLab(image):
    tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
    image = image / np.max(image)
    if image.shape[2] == 1:
        tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
        tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
    else:
        tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229  # r
        tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224  # g
        tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225  # b

    tmpImg = tmpImg.transpose((2, 0, 1))

    return torch.from_numpy(tmpImg.copy())

# 测试图片文件夹路径
images_path = r'D:\DeepLearning'
images_list = os.listdir(images_path)
# 转好的 onnx 权重路径
onnx_model = r"D:\DeepLearning\u2netp.onnx"
# 加载权重
net = cv.dnn.readNetFromONNX(onnx_model)

for image_name in images_list:
    image_path = os.path.join(images_path, image_name)
    img_cv = cv.imdecode(np.fromfile(image_path, dtype=np.uint8), cv.IMREAD_COLOR)

    image = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)
    # (h, w)
    img = transform.resize(image, (1783, 2534), mode='constant')
    sample = ToTensorLab(img)
    inputs_test = sample.unsqueeze(0)
    inputs_test = inputs_test.type(torch.FloatTensor)
    img = inputs_test.numpy()

    net.setInput(img)
    time_start = time.time()
    # 指定 onnx 的输出节点，这个在使用 torch2onnx.py 转 onnx 时可以打印查看
    out = net.forward('1810')
    time_stop = time.time()
    print('inference time cost', time_stop - time_start)
    predict_np = out.squeeze()

    predict_np = cv.resize(predict_np, (img_cv.shape[1], img_cv.shape[0]), interpolation=cv.INTER_NEAREST)
    # rgb
    cls = dict([(1, (128, 0, 128)),
                (2, (255, 255, 0))])

    r = predict_np.copy()
    b = predict_np.copy()
    g = predict_np.copy()
    for c in cls:
        r[r == c] = cls[c][0]
        g[g == c] = cls[c][1]
        b[b == c] = cls[c][2]
    rgb = np.zeros((img_cv.shape[0], img_cv.shape[1], 3))
    print('classes ', np.unique(predict_np))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    # 推理结果保存
    Image.fromarray(rgb.astype(np.uint8)).save('./test_out/' + image_name)
