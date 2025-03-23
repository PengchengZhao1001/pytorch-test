import numpy as np
import matplotlib.pyplot as plt

def DataSet():
    ax=np.random.normal(-3,1,size=(10,1))
    ay=np.random.normal(2,0.5,size=(10,1))
    a=np.hstack((ax,ay))
    bx = np.random.normal(-4, 1, size=(10, 1))
    by = np.random.normal(0, 0.5, size=(10, 1))
    b=np.hstack((bx,by))
    cx = np.random.normal(2, 1, size=(10, 1))
    cy = np.random.normal(2, 0.5, size=(10, 1))
    c = np.hstack((cx, cy))
    dx = np.random.normal(3, 1, size=(10, 1))
    dy = np.random.normal(0, 0.5, size=(10, 1))
    d = np.hstack((dx, dy))
    plt.figure(1)
    plt.subplot(1,2,1)  #一行两列 第一个图像
    plt.plot(ax, ay, 'ro',bx, by, 'bo',cx, cy, 'co',dx, dy, 'mo')
    X = np.concatenate((a,b,c,d),axis=0)
    np.random.shuffle(X)
    return X

def Centroid_initial(X,k):
    #质心初始化
    a=np.random.permutation(X.shape[0])  #x:m*n,m个
    centroids=X[a[:k],:]  #取前K个作为质心
    print("初始质心的数组大小:",centroids.shape)

    return centroids

def K_Means_classifier(X,k,centroids):
    m,n = X.shape
    idex = np.zeros(m,dtype=int)
    #首先计算初始分类
    for i in range(m):
        temp = np.sum((X[i] - centroids[0]) ** 2) #减去一开始的质心
        temp_idex = 0
        for j in range(1,k):
            if np.sum((X[i]-centroids[j])**2) < temp:
                temp = np.sum((X[i]-centroids[j])**2)
                temp_idex = j
        # print(temp_idex)
        idex[i]=temp_idex  # 每一个数据进行分类

    return idex

def K_Means(X,k,idex,centroids_old):
    m, n = X.shape
    # print("数据一开始的类别:",idex)
    centroids_changed = True
    #开始K均值无监督聚类,首先重新找质心,在分类
    while centroids_changed:
        centroids_new = np.zeros((k, n))
        for j in range(k):
            centroids_new_index = np.where(idex == j)[0]
            centroids_new[j] = np.sum(X[centroids_new_index],axis=0)/len(centroids_new_index)
            # counter = 0
            # for i in range(m):
            #     if idex[i] == j:  # 找每个数据的类别信息,进行质心寻找
            #         centroids_new[j] += X[i]
            #         counter += 1
            # centroids_new[j] = centroids_new[j] / counter

        if (centroids_new == centroids_old).all():  # 如果质心不变
            centroids_changed = False
        else:
            # 否则,新的质心,重新进行聚类
            idex = K_Means_classifier(X, k, centroids_new)
            centroids_old = centroids_new
    # print("最终数据的类别:",idex)

    # 当质心不改变时,将类别为i的数据挑出来,归为一类
    temp_class_X = []
    for i in range(k):
        class_index = np.where(idex == i)[0]  #tuple
        temp_class_X.append(X[class_index])

    plt.figure(1)
    plt.subplot(1,2,2)
    for i,color in enumerate("rbcm"):
        plt.plot(temp_class_X[i][:, 0], temp_class_X[i][:, 1], color+'o')
    # plt.plot(temp_class_X[0][:,0],temp_class_X[0][:,1],'ro')
    # plt.plot(temp_class_X[1][:, 0], temp_class_X[1][:, 1], 'bo')
    # plt.plot(temp_class_X[2][:, 0], temp_class_X[2][:, 1], 'co')
    # plt.plot(temp_class_X[3][:, 0], temp_class_X[3][:, 1], 'mo')
    plt.show()


'''运行本py文件代码'''
if __name__ == '__main__':
    Original_data = DataSet()
    centroids_initial = Centroid_initial(Original_data,k=4)
    idex_initial = K_Means_classifier(Original_data,k=4,centroids=centroids_initial)
    K_Means(Original_data,k=4,idex=idex_initial,centroids_old=centroids_initial)








