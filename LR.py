import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

'''纯python,用梯度下降法推导逻辑回归'''

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
    x_data = torch.from_numpy(np_data[:, 0:2])
    # print(x_data.size())
    y_data = torch.from_numpy(np_data[:, -1]).unsqueeze(1)
    # print(y_data.size())
    # y_data = torch.from_numpy(np_data[:, -1])
    return x_data.numpy(), y_data.numpy()

def sigmoid(x):
    return 1/(1+np.exp(-x))

def LR(lr=1e-4, k=1e-8,epoch_n=200000):
    x, y = load_DataSet()
    a, c = x.shape
    w0 = np.zeros([c, 1])
    b0 = 0
    z0 = np.dot(x,w0) + b0
    L0 = np.sum(y * z0 - np.log(1 + np.exp(z0)))
    for epoch in range(epoch_n):
        z = np.dot(x,w0) + b0
        out = sigmoid(z)  # a*1
        mask = torch.from_numpy(out).ge(0.5).float()  # 如果比0.5大，则返回1，如果小就返回0
        correct = torch.sum(mask == torch.from_numpy(y))
        '''梯度下降法'''
        w = w0 + lr * np.dot(x.T,(y-out))  #c*1
        b = b0 + lr* np.sum(y-out)  #a*1-->1
        L1 = np.sum(y * (np.dot(x,w) + b) - np.log(1 + np.exp(np.dot(x,w) + b)))
        if abs(L1-L0)<k:
            print("经历了[%d]epoch,准确率为:[%.4f%%]" % (epoch+1,100*correct/a))
            break
        else:
            L0 = L1
            w0 = w
            b0 = b
            if (epoch + 1) % 100 == 0:
                print("Epoch:[%d],准确率为:[%.4f%%]" % (epoch+1,100*correct/a))
    return w0[0][0],w0[1][0],b0

w0,w1,b = LR()
plt_data()
plot_x=np.arange(30,100,0.1)
plot_y=(-w0*plot_x-b)/w1
plt.plot(plot_x,plot_y)
plt.show()










