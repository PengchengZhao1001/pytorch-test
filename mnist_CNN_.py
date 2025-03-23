import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
#from torch.autograd import variable
import matplotlib.pyplot as plt



transform=transforms.ToTensor()
data_train = datasets.MNIST(root="./data",
                            train=True,
                            transform=transform,
                            download=True)
data_test = datasets.MNIST(root="./data",
                           transform=transform,
                           train=False)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=64,
                                                shuffle=True)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=64,
                                               shuffle=True)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1=torch.nn.Sequential(

            torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.dense=torch.nn.Sequential(
            torch.nn.Linear(14*14*128,1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(1024,10)
        )

    def forward(self, x):
        x=self.conv1(x)
        x=x.view(-1,14*14*128)
        x=self.dense(x)
        return x


model=Model()
print(model.__module__)
model.cuda()
cost=torch.nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(model.parameters())

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
        _,pred = torch.max(outputs.data,1)
        # print("pred",pred)#64*1
        optimizer.zero_grad()
        print(Y_train)
        loss = cost(outputs,Y_train)

        loss.backward()
        optimizer.step()

        #print('',loss.data[0])
        running_loss +=loss.data
        running_correct +=torch.sum(pred == Y_train.data)

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
