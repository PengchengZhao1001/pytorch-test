import math

'''定义函数'''
def function(x,y):
    f = math.exp(x**2 + (y-2)**2)
    return f
def GD(x0,y0,lr=0.0001,k=1e-10,iter=100000,epoch=0):
    print("初始点为(%d, %d)" % (x0,y0))
    f0 = function(x0, y0)
    # print(f0)
    for i in range(iter):
        y1 = y0-lr*(function(x0,y0) * 2 * (y0-2))
        x1 = x0-lr*(function(x0,y0) * 2 * x0)
        f1 = function(x1,y1)
        print("第%d次迭代结果为(%.4f, %.4f, %.4f)" % (i,f1,x1,y1))
        if abs(f1-f0) < k:
            epoch = i
            break
        else:
            f0 = f1
            y0 = y1
            x0 = x1
    print("*"*20)
    print("经过%d次的迭代, 得最优点为(%.4f, %.4f)" % (epoch,x1,y1))
GD(x0=1,y0=1)





# a=2
# b=(a-1)**2
# print(b)