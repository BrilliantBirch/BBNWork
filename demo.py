import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tensorboardX import SummaryWriter
import numpy as np

# 准备数据
x = np.random.rand(100, 1)
y = 2 * x + 1 + 0.1 * np.random.rand(100, 1)
dataset = TensorDataset(torch.Tensor(x), torch.Tensor(y))
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 定义简单的线性回归模型
class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleLinearModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 初始化 SummaryWriter
writer = SummaryWriter('runs/simple_linear_model')

# 训练模型
for epoch in range(100):
    for i, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # 每个 epoch 记录一次 loss
    writer.add_scalar('Loss/train', loss.item(), epoch)
    
    # 每10个epoch记录一次模型参数
    if epoch % 10 == 0:
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

    print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

writer.close()

print("Training complete. Check the 'runs/simple_linear_model' directory for the logs.")
