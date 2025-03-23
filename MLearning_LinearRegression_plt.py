'''最简单的一维线性回归'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

x_train=np.array([[3.3],[4.4],[5.5],[6.71],[6.93],[4.168],
                  [9.779],[6.182],[7.59],[2.167],[7.042],
                  [10.791],[5.313],[7.997],[3.1]],dtype=np.float32)
y_train=np.array([[1.7],[2.76],[2.09],[3.19],[1.694],[1.573],
                  [3.366],[2.596],[2.53],[1.221],[2.827],
                  [3.465],[1.65],[2.904],[1.3]],dtype=np.float32)
'''
#绘制散点图显示
plt.scatter(x_train,y_train)
plt.xlabel("x_train")
plt.ylabel("y_train")
plt.show()
'''
x_train = torch.from_numpy(x_train)
y_train=torch.from_numpy(y_train)

'''建立模型'''
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        out=self.linear(x)
        return out


model=LinearRegression()
cost=torch.nn.MSELoss()
optimizer= torch.optim.Adam(model.parameters())

'''训练模型'''
epoch_n=1000
for epoch in range(epoch_n):
    inputs=Variable(x_train)
    target=Variable(y_train)
    out=model(inputs)
    loss=cost(out,target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1)%20 == 0:
        print("Epoch[{}/{}],loss:{:.4f}".format(epoch+1,epoch_n,loss.data[0]))

'''测试部分'''
model.eval()
#将模型变成测试模式，因为有一些层操作，比如dropout、batchnormalization在训练和测试时不一样
prdict =model(inputs)
prdict=prdict.data.numpy()
plt.plot(x_train.numpy(),y_train.numpy(),'ro')
plt.plot(x_train.numpy(),prdict,'k-')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()