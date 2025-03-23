import torch

#先定义输入、输出、隐藏层的特征数
batch_n = 100 #每个批次有100个数据
hidden_layer = 100 #隐层后保留的特征数
input_data=1000 #每个数据有1000个特征
output_data=10 #输出有10个分类结果

#再去定义层与层之间的初始化权重参数
x=torch.randn(batch_n,input_data) #定义输入层维度
y=torch.randn(batch_n,output_data) #定义输出层维度
w1=torch.randn(input_data,hidden_layer)
w2=torch.randn(hidden_layer,output_data)

#明确训练次数和学习率
epoch_n=20
learning_rate=1e-4

#使用梯度下降法后向传播
for epoch in range(epoch_n):
    h1=x.mm(w1)#100*100
    h1=h1.clamp(min=0) #类似激活函数
    y_pred=h1.mm(w2) #100*10

    loss=(y_pred-y).pow(2).sum()/batch_n#均方误差函数
    print("Epoch:{},loss:{:.4f}".format(epoch,loss))

    #一次结束后更新参数
    grad_loss=2*(y_pred-y)/batch_n
   # print("loss求导",grad_loss)
    grad_y_pred=h1
    grad_w2=grad_y_pred.mm(grad_loss)

    grad_h=grad_loss.mm(w2.t())
    grad_h.clamp_(min=0)
    grad_w1=x.t().mm(grad_h)

    w1 -=learning_rate*grad_w1
    w2 -=learning_rate*grad_w2