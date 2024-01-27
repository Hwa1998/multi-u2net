import torch
from torch.autograd import Variable
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

import numpy as np
import glob
import os

from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensorLab
from data_loader import SalObjDataset

import random

from model.u2net import U2NET, U2NETP


# ------- 0. set random seed --------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print("---set seed...")

set_seed(1000)
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
# ------- 1. define loss function --------
# CrossEntropyLoss
device = torch.device("cuda:0")
# background 的权重是 1 ，其他类别的默认为 1.5 也没问题，有个 classes weight 的计算方式的帖子可以参考
#           https://blog.csdn.net/magic_ll/article/details/123377662
# 下面有个 定义模型 的代码，第二个通道数是 类别数量(背景+分割类别)，这个记得改
weights = np.array([1.00, 1.50, 1.50], dtype=np.float32)
weights = torch.from_numpy(weights).to(device)
loss_CE = nn.CrossEntropyLoss(weight=weights).to(device)


def muti_CrossEntropyLoss_loss(d0, d1, d2, d3, d4, d5, d6, labels_v):
    labels_v = labels_v.squeeze(1).long()
    loss0 = loss_CE(d0, labels_v)
    loss1 = loss_CE(d1, labels_v)
    loss2 = loss_CE(d2, labels_v)
    loss3 = loss_CE(d3, labels_v)
    loss4 = loss_CE(d4, labels_v)
    loss5 = loss_CE(d5, labels_v)
    loss6 = loss_CE(d6, labels_v)

    # loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6  # u2net5p
    loss = loss0 * 1.5 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"
          % (loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(),
             loss4.data.item(), loss5.data.item(), loss6.data.item()))

    return loss0, loss


# ------- 2. set the directory of training process --------
model_name = 'u2netp'  # 'u2net'
model_dir = os.path.join('saved_models', model_name + os.sep)

# 图片的文件类型
image_ext = '.png'
label_ext = '.png'

# 训练集集路径
data_dir = os.path.join("datasets/train_data" + os.sep)
# 原始图片路径
tra_image_dir = os.path.join('images' + os.sep)
# 图片的标签路径，这里读到的图片的 rgb 值分别为 0、1、2、。。。，是位深度为 8 的单通道掩码图
tra_label_dir = os.path.join('masks' + os.sep)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 训练回合
epoch_num = 10

batch_size = 16
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
for img_path in tra_img_name_list:
    img_name = img_path.split(os.sep)[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        # resize to 512*512
        RescaleT(512),
        # random crop
        RandomCrop(488),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

# ------- 3. define model --------
# define the net
if model_name == 'u2net':
    #          img channel, classes number(background + classes)
    net = U2NET(3, 3)
elif model_name == 'u2netp':
    net = U2NETP(3, 3)
    # net.load_state_dict(torch.load(seg_pretrain_u2netp_path, map_location=torch.device("cpu")))

if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)  # u2netp、u2net2p

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
# save_frq = num_train_data / batch size
save_frq = 3000

for epoch in range(0, epoch_num):
    net.train()
    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_CrossEntropyLoss_loss(d0, d1, d2, d3, d4, d5, d6, labels_v)
        loss.backward()
        # for param in net.parameters():
        #     param.goptimizer.step()rad = None
        # nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()

        # print statistics
        running_loss += loss.data.item()
        running_tar_loss += loss2.data.item()

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f "
              % (epoch + 1, epoch_num, (i + 1) * batch_size, train_num, ite_num, running_loss
                 / ite_num4val, running_tar_loss / ite_num4val))

        if ite_num % save_frq == 0:
            torch.save(net.state_dict(), model_dir + model_name + "_bce_itr_%d.pth" % ite_num)
            model_path_save = model_dir + model_name + "_bce_itr_" + str(ite_num) + ".pth"
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0
