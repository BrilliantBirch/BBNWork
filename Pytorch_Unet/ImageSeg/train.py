import torch.utils
from model.unet_model import UNet
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch

def train_net(net,device,data_path,epochs=40,batch_size=1,lr=0.00001):
    """网络训练

    Args:
        net (_type_): _description_
        device (_type_): _description_
        data_path (_type_): _description_
        epochs (int, optional): _description_. Defaults to 40.
        batch_size (int, optional): _description_. Defaults to 1.
        lr (float, optional): _description_. Defaults to 0.00001.
    """
    #加载训练集
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)
    #定义RMSporp算法
    #lr-学习率 weight-decay-L2正则化（权重衰减） momentum-动量
    optimizer = optim.RMSprop(net.parameters(),lr=lr,weight_decay=1e-8,momentum=0.9)
    #定义损失函数
    criterion = nn.BCEWithLogitsLoss()
    #best-loss统计 初始化为正无穷
    best_loss = float('inf')
    
    for epoch in range(epochs):
        #训练模式
        net.train()
        #按照batch_size开始训练
        for image,label in train_loader:
            #PyTorch 默认会累积梯度，因此在每次进行反向传播之前，需要先将梯度清零，
            # 否则之前计算的梯度会影响当前梯度的计算,
            # 导致参数更新不正确
            optimizer.zero_grad()
            #将数据拷贝到device
            image = image.to(device=device,dtype = torch.float32)
            label = label.to(device =device,dtype = torch.float32)
            #使用网络参数，输出预测结果
            pred = net(image)
            #计算loss
            loss = criterion(pred,label)
            print('loss/train',loss.item())
            if loss<best_loss:
                best_loss = loss
                #保存loss值最小的网络参数
                torch.save(net.state_dict(),'best_model.pth')
            #更新参数
            loss.backward()
            optimizer.step()
if __name__ == '__main__':
    #选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #加载网络
    net = UNet(n_channels=1,n_classes=1)
    #将网络拷贝到device
    net.to(device=device)
    #指定训练集地址
    data_path = "Pytorch_Unet/ImageSeg/data/train/"
    train_net(net,device,data_path)    
    