import cv2 as cv
import numpy as np
import onnxruntime as rt
import time
import torch
from PIL import Image
from skimage import transform


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

# 测试图片路径
image_path = ''
img_cv = cv.imdecode(np.fromfile(image_path, dtype=np.uint8), cv.IMREAD_COLOR)

image = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)
# h, w
img = transform.resize(image, (1783, 2534), mode='constant')
sample = ToTensorLab(img)
inputs_test = sample.unsqueeze(0)
inputs_test = inputs_test.type(torch.FloatTensor)
img = inputs_test.numpy()
# sess = rt.InferenceSession("saved_models/u2netp/u2netp.onnx", providers=["CUDAExecutionProvider"])
sess = rt.InferenceSession("saved_models/u2netp/u2netp.onnx", providers=["CPUExecutionProvider"])

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
time_start = time.time()
result = sess.run([output_name], {input_name: img})
time_end = time.time()
print('inference time cost', time_end - time_start)

pred = result[0]
predict_np = pred.squeeze()
predict_np = cv.resize(predict_np, (img_cv.shape[1], img_cv.shape[0]), interpolation=cv.INTER_NEAREST)
# rgb
cls = dict([(1, (0, 255, 0)),
            (2, (255, 0, 0)),
            (3, (255, 0, 255)),
            (4, (255, 255, 0)),
            (5, (0, 0, 255))])
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
# Image.fromarray(rgb.astype(np.uint8)).save('results_onnx.png')
