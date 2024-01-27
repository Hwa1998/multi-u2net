import cv2
import torch
from torch.autograd import Variable

import numpy as np
from PIL import Image

from data_loader import RescaleT, Rescale
from data_loader import ToTensorLab

from model.u2net import U2NET, U2NETP

from skimage import io
import os
import time


def eval_main(model, sample, img_size):
    for i_test, data_test in enumerate([sample]):
        inputs_test = data_test['image']

        inputs_test = inputs_test.unsqueeze(0)
        inputs_test = inputs_test.type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)
        t1 = time.time()
        d1, d2, d3, d4, d5, d6, d7 = model(inputs_test)
        t2 = time.time()
        print(t2-t1)
        d1 = torch.softmax(d1, dim=1)
        predict_np = torch.argmax(d1, dim=1, keepdim=True)
        predict_np = predict_np.cpu().detach().numpy().squeeze()

        predict_np = cv2.resize(predict_np, img_size, interpolation=cv2.INTER_NEAREST)

        cls = dict([(1, (0, 0, 255)),  # rgb
                    (2, (255, 0, 255)),
                    (3, (0, 255, 0)),
                    (4, (255, 0, 0)),
                    (5, (255, 255, 0))])


        r = predict_np.copy()
        b = predict_np.copy()
        g = predict_np.copy()
        for c in cls:
            r[r == c] = cls[c][0]
            g[g == c] = cls[c][1]
            b[b == c] = cls[c][2]

        rgb = np.zeros((img_size[1], img_size[0], 3))
        print('classes', np.unique(predict_np))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        Image.fromarray(rgb.astype(np.uint8)).save('test_out/test_out.png')
        del d1, d2, d3, d4, d5, d6, d7


def label_deal(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])
    if 3 == len(label_3.shape):
        label = label_3[:, :, 0]
    elif 2 == len(label_3.shape):
        label = label_3
    if 3 == len(image.shape) and 2 == len(label.shape):
        label = label[:, :, np.newaxis]
    elif 2 == len(image.shape) and 2 == len(label.shape):
        label = label[:, :, np.newaxis]
    return label


if __name__ == "__main__":
    # # --------- 1. get image and model ---------
    # pth 权重路径
    model_dir = 'saved_models/u2netp.pth'
    # 测试图片路径
    image = io.imread('')
    # 推理结果的保存路径
    if not os.path.exists('test_out/'):
        os.makedirs('test_out/')

    # --------- 2. pre-deal data ---------
    label = label_deal(image)
    if 2 == len(image.shape) and 2 == len(label.shape):
        image = image[:, :, np.newaxis]
    imidx = np.array([0])

    sample = {'imidx': imidx, 'image': image, 'label': label}
    deal1 = RescaleT(512)
    deal2 = ToTensorLab(flag=0)
    sample = deal1(sample)
    sample = deal2(sample)

    # --------- 3. model define ---------
    net = U2NETP(3, 3)
    model = torch.load(model_dir)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    # --------- 4. model infer ---------
    model = net.eval()
    eval_main(model, sample, (image.shape[1], image.shape[0]))
