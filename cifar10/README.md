# cifar10训练
## ResNet
按照论文描述构建BasicBlock、BottlenBlock；ResNet18、ResNet34由BasicBlock构建；ResNet50、ResNet101、ResNet152由BottlenBlock构成
> 按照论文实现的ResNet50在cifar10上进行训练，数据预处理仅作std处理；训练100个epoch，准确率收敛至88%左右；考虑需要对网络结构进行优化、增加数据预处理操作增强数据
## 改进
1. 除了最后的平均池化和全连接层，其余全部改为卷积层(删去原始网络的最大池化层)

2. 由于cifar10的图片大小为32$\times$32，将第一层卷积核大小改为3$\times$3，步长改为1，填充改为1；每层channel大小不变

3. 数据预处理：增加随机裁剪(transforms.RandomCrop)和随机水平翻转(transforms.RandomHorizontalFlip())

4. 训练的超参数调整：使用交叉熵作为训练损失，使用带动量的SGD作为优化算法，使用余弦退火策略调整优化器的学习率








