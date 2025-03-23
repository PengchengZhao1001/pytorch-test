import torch
from torch.autograd import Variable  #引入类

batch_n = 100
hidden_layer = 100
input_data = 1000
output_data = 10

x = Variable(torch.randn(batch_n,input_data),requires_grad=False)
y = Variable(torch.randn(batch_n,output_data),requires_grad=False)
w1 = Variable(torch.randn(input_data,hidden_layer),requires_grad=True) #该变量在进行自动梯度计算的过程中保留梯度值
w2 = Variable(torch.randn(hidden_layer,output_data),requires_grad=True)

epoch_n = 10000
learning_rate = 1e-7

for epoch in range(epoch_n):
    h1=x.mm(w1).clamp(min=0)
    y_pred = h1.mm(w2)
    loss = (y_pred-y).pow(2).sum()
    print(loss)
    if epoch%1000 == 0:
        print("Epoch:{},loss:{:.4f}".format(epoch,loss.data[0]))

    loss.backward()

    w1.data -= learning_rate*w1.grad.data
    w2.data -= learning_rate*w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()