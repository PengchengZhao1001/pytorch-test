import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
#from torch.autograd import variable
import matplotlib.pyplot as plt
# torchvision包主要功能是 实现数据的处理、导入和预览

'''接下来引入数据集的测试集和训练集'''
transform=transforms.ToTensor()
data_train = datasets.MNIST(root="./data",  # 此处根目录为该文件夹下
                            train=True,
                            transform=transform,
                            download=True)
data_test = datasets.MNIST(root="./data",
                           transform=transform,
                           train=False)
# #print(len(data_train))
#
# '''数据装载（打包给我们的模型）'''
data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=64,  # 每个包中的数据个数
                                                shuffle=True)  # 装载过程中打乱顺序

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=64,
                                               shuffle=True)
#
#
# images, labels = next(iter(data_loader_train))
# img=torchvision.utils.make_grid(images)#将一个批次的图片构造成网格模式,改变维度（4维变为三维）
#
# img=img.numpy().transpose(1,2,0)
# std=[0.5,0.5,0.5]
# mean=[0.5,0.5,0.5]
# img=img*std+mean
# print([labels[i] for i in range(64)])
# plt.imshow(img)
# plt.show()


'''模型的搭建和参数的优化'''
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1=torch.nn.Sequential(
            #因为mnist数据集是灰度图，所以一张图片的通道数为1
            torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2)#尺寸变为一半
        )
        '''定义全连接层'''
        self.dense=torch.nn.Sequential(
            torch.nn.Linear(14*14*128,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024,10)
        )

    def forward(self, x):
        x=self.conv1(x)
        x=x.view(-1,14*14*128)   #对参数进行扁平化
        x=self.dense(x)
        return x


model=Model()
print(model.__module__)
model.cuda()
cost=torch.nn.CrossEntropyLoss()
'''学习率默认为0.001'''
optimizer=torch.optim.Adam(model.parameters())
'''模型训练'''
epoch_n=1
#
for epoch in range(epoch_n):
    running_loss = 0.0
    running_correct=0
    print("Epoch {}/{}".format(epoch,epoch_n))
    print("-"*10)
   # for data in data_loader_train:
        #X_train,Y_train = data
        #print(Y_train)#64*1
    for batch_idx,(X_train,Y_train) in enumerate(data_loader_train):
        print(Y_train)
        X_train=X_train.cuda()
        Y_train=Y_train.cuda()
        X_train,Y_train = Variable(X_train),Variable(Y_train)
        outputs = model(X_train)
        # print(outputs)#64*10
        _,pred = torch.max(outputs.data,1)  #取最大的对应的标签,是标签也就是第几个，前面一个_取的是最大值，是数值
        # print("pred",pred)#64*1
        optimizer.zero_grad()
        print(Y_train)
        loss = cost(outputs,Y_train)

        loss.backward()
        optimizer.step()

        #print('',loss.data[0])
        running_loss +=loss.data #为什么要累加
        running_correct +=torch.sum(pred == Y_train.data)#准确率

    testing_correct=0
    for data in data_loader_test:
        X_test, Y_test = data
        X_test=X_test.cuda()
        Y_test=Y_test.cuda()
        X_test, Y_test = Variable(X_test), Variable(Y_test)
        outputs = model(X_test)
        _,pred=torch.max(outputs.data,1)
        testing_correct +=torch.sum(pred == Y_test.data)

    print("Loss is:{:.4f},Train Accuracy is:{:.4f}%,Test Accuracy is:{:.4f}%".format(running_loss/len(data_train),
                                                                                     100*running_correct/len(data_train),
                                                                                     100 * testing_correct/len(data_test)))
# '''
# data_loader_test =torch.utils.data.DataLoader(dataset=data_test,batch_size=4,shuffle=True)
# X_test,Y_test=next(iter(data_loader_test))
# X_test = X_test.cuda()
# Y_test = Y_test.cuda()
# inputs = Variable(X_test)
# pred = model(inputs)
# #print("输出为：",pred.data)
# _,pred=torch.max(pred,1)
# print("Predict Lable is:",[i for i in pred.data])
# print("Real Lable is:",[i for i in Y_test])
#
# img =torchvision.utils.make_grid(X_test)
# img=img.cpu()
# img = img.numpy().transpose(1,2,0)
#
# std=[0.5,0.5,0.5]
# mean=[0.5,0.5,0.5]
# img=img*std+mean
# plt.imshow(img)
# plt.show()
# '''