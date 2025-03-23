import torch
from torch.autograd import Variable

batch_n = 100
input_data = 1000
output_data =10
hidden_layer = 100

x = Variable(torch.randn(batch_n,input_data),requires_grad=False)
y = Variable(torch.randn(batch_n,output_data),requires_grad=False)
#无需定义权重参数，由torch.nn中的类自动定义了，给了更好的初始化

models = torch.nn.Sequential(
    torch.nn.Linear(input_data,hidden_layer),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_layer,output_data)
)

epoch_n = 10000
learning_rate =1e-4
loss_forward = torch.nn.MSELoss()#建立一个对象
#自动优化参数，自适应调节学习率
optimzer = torch.optim.Adam(models.parameters(),lr=learning_rate)
for epoch in range(epoch_n):
    y_pred = models(x)
    loss = loss_forward(y_pred,y)
    if epoch%1000 == 0:
        print("Epoch:{},loss:{:.4f}".format(epoch,loss.data))
   # models.zero_grad()
    optimzer.zero_grad()

    loss.backward()

    '''for param in models.parameters():
        param.data -= param.grad.data*learning_rate
        #对模型中的全部参数进行进行遍历
    '''
    optimzer.step()