'''
代码运行说明：（图文说明请查看试验报告）
1.首先需要安装Pytorch库，matplotlib库
2.将所要比较的两张图片放入'faces_data/testing/t'目录中
3.如果需要比较多个数据文件夹，可将代码中的##9.1包含内容全部注释，将##9.2中的内容取消注释
  然后将所需比较的批量图片文件放入'faces_data/testing’目录中
'''
import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import PIL.ImageOps

##1.前期准备，定义超参数、需要用到的函数
train_batch_size = 32  # 训练时batch_size
train_number_epochs = 50  # 训练的epoch
print("Batch_size:{0}   Epochs_num:{1}".format(train_batch_size, train_number_epochs))


def imshow(img, text=None, should_save=False):
    # 展示一幅tensor图像，输入是(C,H,W)
    npimg = img.numpy()  # 将tensor转为ndarray
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8,
                       'pad': 10})  # plt.text（）函数，（75，8）是位置，bbox设置边框，facecolor-背景颜色，alpha-透明度，pad-填充
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 转换为(H,W,C)，因为imshow输入参数为HWC。transpose为转置函数（X轴=0；Y轴=1；Z轴=2）
    plt.show()  # imshow函数仅对图片进行处理，show函数才展示出来


def show_plot(iteration, loss):
    # 绘制损失变化图
    plt.plot(iteration, loss)
    plt.show()


##2.准备数据
# 自定义Dataset类，__getitem__(self,index)每次返回(img0, img1, 0/1)  0表示同一类的图片，1表示不是同一类
class SiameseNetworkDataset(Dataset):  # 重写父类Dataset的函数

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(
            self.imageFolderDataset.imgs)  # 37个类别中任选一个,取出的是该类的标签，比如说s1文件中的所有图片的标签都被dataloader自动设置为了0
        should_get_same_class = random.randint(0, 1)  # 保证同类样本约占一半
        if should_get_same_class:
            while True:
                # 直到找到同一类别
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:  # img0_tuple[0]是该图片的相对地址，img0_tuple[1]是该图片的类别标签
                    break
        else:
            while True:
                # 直到找到非同一类别
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])  # open只是打开该目录下的图片，并不展示，show()才展示
        img0 = img0.convert("L")  # 转换为L mode，即将图片转为灰度图,即黑白图。好处在于可以忽略颜色带来的影响，使网络专注于人脸特征的提取
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)  # invert黑白颜色反转,可以通过Image._show(img0)分别展示
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:  # 如果transfor不为空，就执行对应的transform操作，对图片进行处理
            img0 = self.transform(img0)  # 使用传入的transform实例，对图片进行处理
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


# 定义文件dataset
training_dir = "faces_data/training"  # 训练集地址
folder_dataset = torchvision.datasets.ImageFolder(root=training_dir)  # 读取图片，实例化一个ImageFolder对象，读取按照指定文件夹的目录格式所保存的图片

# 定义图像dataset
transform = transforms.Compose([transforms.Resize((100, 100)),  # 传入int和tuple有区别
                                transforms.ToTensor()])  # 先裁剪为100*100，再转化为tensor（张量）形式
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transform,
                                        should_invert=False)  # 不需要进行invert（灰度图0，1互换）

# 定义图像dataloader(训练集数据加载器)——DataLoader类的实例，可以自动将图片数据分成批，还能随机打乱顺序
train_dataloader = DataLoader(siamese_dataset,
                              shuffle=True,
                              batch_size=train_batch_size)  # shuffle：在每个epoch开始的时候，对数据进行重新打乱
print('训练数据加载完毕')


##3.搭建网络模型
class SiameseNetwork(nn.Module):  # nn.Module是Pytorch提供的强大的父类，其中包含了绝大部分关于神经网络的通用计算，其中__init()__等部分函数可以重写
    def __init__(self):
        super(SiameseNetwork, self).__init__()  # 先调用父类的构造函数，再构造SiameseNetwork网络用到的神经模块
        self.cnn1 = nn.Sequential(  # nn.Sequential是一个有序的容器，神经网络模块会按照传入构造器的顺序依次执行，使用Sequential构造网络，比传统的构造方式更好
            nn.ReflectionPad2d(1),
            # 使用一个数值，进行边界镜像填充，意思就是，只填充一层（只在外部包裹一层数据），这个数据来自已有数据中的镜像值（很简单，测试一下就懂了）。目的在于卷积过后还是100*100图像
            nn.Conv2d(1, 4, kernel_size=3),  # 定义一个卷积层，输入通道为1（即灰度图，上文有对输入图片做处理），输出通道为4，卷积核窗口大小为3*3
            nn.ReLU(inplace=True),
            # ReLU是激活函数，效果比传统的sigmod函数好。inplace参数的作用是：是否将改变，作用到原数据。如果inplace=true，则进行覆盖运算，即对原值操作，并用结果替代原值，而不产生结果的copy，节省内存
            nn.BatchNorm2d(4),  # 对数据进行归一化处理，参数为num_features(通道数)

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),  # 线性连接层，输入尺寸为8*100*100，输出为500个节点
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):  # forwaed()函数在正向运行神经网络的时候被自动调用，负责数据的向前传递
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


##4.定义ContrastiveLoss
class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)  # 计算两个向量间的欧式距离
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
                                                          2))  # 损失函数计算

        return loss_contrastive


##5.定义训练参数、变量
net = SiameseNetwork().cuda()  # 创建网络的实例且移至GPU进行计算
criterion = ContrastiveLoss()  # 定义损失函数
optimizer = optim.Adam(net.parameters(), lr=0.0005)  # 定义优化器，net.parameters()得到nn.module预设的神经元参数，设置学习率为0.0005

counter = []
loss_history = []
iteration_number = 0

##6.开始训练
for epoch in range(0, train_number_epochs):
    for i, data in enumerate(train_dataloader, 0):
        img0, img1, label = data
        # img0维度为torch.Size([32, 1, 100, 100])，32是batch，label为torch.Size([32, 1])
        img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()  # 数据移至GPU
        optimizer.zero_grad()  # 对模型参数的梯度归0，避免梯度的累加
        output1, output2 = net(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()  # 使用backward（）函数，自动计算得到参数的梯度值
        optimizer.step()  # 使用计算得到的梯度值对各个节点的参数进行梯度更新
        if i % 10 == 0:  # 10次迭代为一个x轴坐标
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
    print("Epoch number: {} , Current loss: {:.4f}\n".format(epoch, loss_contrastive.item()))

print('训练结束')
show_plot(counter, loss_history)

##7.定义测试集的dataset和dataloader

# 定义文件dataset
testing_dir = "faces_data/testing"  # 测试集地址
folder_dataset_test = torchvision.datasets.ImageFolder(root=testing_dir)

# 定义图像dataset
transform_test = transforms.Compose([transforms.Resize((100, 100)),
                                     transforms.ToTensor()])
siamese_dataset_test = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                             transform=transform_test,
                                             should_invert=False)

# 定义图像dataloader
test_dataloader = DataLoader(siamese_dataset_test,
                             shuffle=True,
                             batch_size=1)
print('测试数据加载完毕')

##8.计算在该网络下训练出的最大不相似度。算法：使用invert进行0，1反转，使特征值差距最大，求得特征值的欧式距离，再通过归一化处理
init_img_tuple = random.choice(folder_dataset_test.imgs)
init_img = Image.open(init_img_tuple[0])
init_img = init_img.convert("L")
invert_img = PIL.ImageOps.invert(init_img)
init_img = transform_test(init_img)
init_img = init_img.unsqueeze(0)
invert_img = transform_test(invert_img)
invert_img = invert_img.unsqueeze(0)  # 使用该方法增加维度，使经过前期转化的图片由（1，100，100）→（1，1，100，100）
output1, output2 = net(init_img.cuda(), invert_img.cuda())
euclidean_distance = F.pairwise_distance(output1, output2)
max_dissimilar = euclidean_distance.item()
print("MaxDissimilarity:{0}".format(max_dissimilar))

##9.1计算文件中两张图片的相似度
x0 = Image.open(test_dataloader.dataset.imageFolderDataset.imgs[0][0])
x0 = x0.convert("L")
x0 = transform_test(x0)
x0 = x0.unsqueeze(0)
x1 = Image.open(test_dataloader.dataset.imageFolderDataset.imgs[1][0])
x1 = x1.convert("L")
x1 = transform_test(x1)
x1 = x1.unsqueeze(0)
concatenated = torch.cat((x0, x1), 0)
output1, output2 = net(x0.cuda(), x1.cuda())
euclidean_distance = F.pairwise_distance(output1, output2)
imshow(torchvision.utils.make_grid(concatenated),
       'Similarity: {:.2f}%'.format(((max_dissimilar - euclidean_distance.item()) / max_dissimilar) * 100))

# ##9.2利用模型对测试集进行相似度计算，并生成对比图像
# dataiter = iter(test_dataloader)  # 获取test_dataloader的迭代器
# for i in range(10):
#     x0, x1, label2 = next(dataiter)  # dataloader的格式是这样：[图片1，图片2，是否为同一类]。原因在于Dataloader的参数siamese_dataset_test返回的值为img0，img1，label
#     concatenated = torch.cat((x0, x1), 0)
#     output1, output2 = net(x0.cuda(), x1.cuda())
#     euclidean_distance = F.pairwise_distance(output1, output2)
#     imshow(torchvision.utils.make_grid(concatenated), 'Similarity: {:.2f}%'.format(((max_dissimilar-euclidean_distance.item())/max_dissimilar)*100))
