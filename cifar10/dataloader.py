import torch
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
batch_size = 100
#Cifar10 dataset                    #选择数据的根目录   #选择训练集    #从网上下载图片
train_dataset = dsets.CIFAR10(root = './dataset', train= True, download= True)
                                    #选择数据的根目录   #选择训练集    #从网上下载图片
test_dataset = dsets.CIFAR10(root = './dataset', train= False, download= True)
#加载数据
#将数据打乱
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle= True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = batch_size,shuffle= True)
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
digit = train_loader.dataset.data[0]
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()
print(classes[train_loader.dataset.targets[0]])

