import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import numpy as np

# with as 作用：获取一个文件句柄，从文件中读取数据，然后关闭文件句柄
with open('logistic_data.txt', 'r')as file:
    data_list = file.readlines()  # readline是每次只存一行，而readlines是所有行存入list中
    data_list = [i.split('\n')[0] for i in data_list]  # 以\n分割，并且取第一个序列，
    data_list = [i.split(',') for i in data_list]  # 再按逗号分隔
    data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]

'''将数据画出来'''
def plt_data():
    # 将0和1分开
    x0 = list(filter(lambda x: x[-1] == 0.0, data))  # lambda为匿名函数，省得命名
    x1 = list(filter(lambda x: x[-1] == 1.0, data))  # filter为内置函数，用于过滤
    plot_x0_x = [i[0] for i in x0]
    plot_x0_y = [i[1] for i in x0]
    plot_x1_x = [i[0] for i in x1]
    plot_x1_y = [i[1] for i in x1]
    plt.plot(plot_x0_x, plot_x0_y, 'ro', label='x_0')
    plt.plot(plot_x1_x, plot_x1_y, 'bo', label='x_1')
    plt.legend(loc='best')
    # plt.show()

def load_DataSet():
    '''构造数据集、标签'''
    np_data = np.array(data, dtype='float32')
    # x_data = torch.from_numpy(np_data[:, 0:2])
    # y_data = torch.from_numpy(np_data[:, -1]).unsqueeze(1)
    # print(y_data.size())
    x_data = np_data[:, 0:2]
    y_data = np_data[:, -1]
    np.random.seed(256)
    np.random.shuffle(y_data)
    np.random.seed(256)
    np.random.shuffle(x_data)
    return torch.from_numpy(x_data), torch.from_numpy(y_data).unsqueeze(1)

class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = torch.nn.Linear(2, 1)
        self.sm = torch.nn.Sigmoid()

    def forward(self, z):
        x_out = self.lr(z)
        x_out = self.sm(x_out)
        return x_out

model = LogisticRegression()
model.cuda()
cost = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-6,momentum=0.9)  # 对于随机梯度下降法不是很理解

'''训练模型'''
Epoch_n = 100000
for epoch in range(Epoch_n):
    dataM, labelM = load_DataSet()
    x = Variable(dataM).cuda()
    y = Variable(labelM).cuda()
    out = model(x)#输出每个点对应的概率
    # print(out)  #确实变化了,但是都是大于0.5的
    mask=out.ge(0.5).float()#如果比0.5大，则返回1，如果小就返回0
    # print(mask)
    correct=torch.sum(mask == y)   # 为啥loss一直下降,准确率却没有,并且维持在64%
    loss = cost(out, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print("Epoch:{},loss:{:.4f},accuracy：{:.4f}%".format(epoch+1,
                                                         loss.data[0],
                                                         100*correct/len(data)))

plt_data()
w0,w1=model.lr.weight[0]
w0=float(w0.cpu().data[0])
w1=float(w1.cpu().data[0])
b=float(model.lr.bias.cpu().data[0])
plot_x=np.arange(30,100,0.1)
plot_y=(-w0*plot_x-b)/w1
plt.plot(plot_x,plot_y)
plt.show()