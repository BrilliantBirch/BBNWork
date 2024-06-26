import torch
from PIL import Image
import os
import glob
import torch.utils
from torch.utils.data import Dataset
import random
import torchvision.transforms as transforms
from PIL import ImageFile

class Garbage_Loader(Dataset):
    """垃圾数据集读取

    
    """
    def __init__(self,text_path,train_flag = True):
        self.imgs_info = self.get_images(text_path)
        self.train_flag = train_flag
        
        self.train_tf = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        
        self.val_tf = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
        
    def get_images(self,text_path):
        """获取图片信息

        Args:
            text_path (str): 

        Returns:
            _type_: _description_
        """
        with open(text_path,'r',encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x:x.strip().split('\t'),imgs_info))
        return imgs_info
    
    def padding_black(self,img):
        """将图片等比例填充到224*224的黑色背景图中

        Args:
            img (_type_): 输入图片

        Returns:
            _type_: 填充后图片
        """
        w,h = img.size
        #计算缩放比
        scale = 224./max(w,h)
        img_fg = img.resize([int(x) for x in [w*scale,h*scale]])
        size_fg = img_fg.size
        size_bg = 224
        img_bg = Image.new('RGB',(size_bg,size_bg))
        img_bg.paste(img_fg,((size_bg-size_fg[0])//2,
                             (size_bg-size_fg[1])//2))
        img = img_bg
        return img
    
    def __getitem__(self, index):
        img_path,label = self.imgs_info[index]
        #调试用路径
        # img = Image.open('ResNet\\'+img_path)
        img = Image.open(img_path)
        
        img = img.convert('RGB')
        img = self.padding_black(img)
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        label = int(label)
        
        return img,label

    def __len__(self):
        return len(self.imgs_info)

if __name__ == "__main__":
    #调试用路径
    # train_dataset = Garbage_Loader(r'ResNet\train.txt')
    #非调试路径
    train_dataset = Garbage_Loader(r'ResNet\train.txt')
    
    print('数据个数：',len(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size =1,
        shuffle = True
    )
    for batch_idx, (image, label) in enumerate(train_loader):
         print(f"Batch {batch_idx}:")
         print("image.cuda:", image.cuda())  # 打印输入数据的形状

         print("Data shape:", image.shape)  # 打印输入数据的形状
         print("label:", label)  # 打印标签数据的形状
         break  # 为了演示，只打印第一个批次的数据
    # for image,label in train_loader:
    #     print(image.shape)
    #     print(label)
    