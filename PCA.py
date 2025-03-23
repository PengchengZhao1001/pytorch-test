import numpy as np
import matplotlib.pyplot as plt

'''PCA,原始数据随机'''
'''原始数据为100个,50个维度,降为2个维度'''
def pca(data, k):
    mean = np.mean(data,axis=0)  # n*m ??为什么不是原始数据是m*n的,m代表维数,n代表样本数  网上都是0;
    centr = data - mean  # 中心化
    cov_data = np.cov(centr.T)  # 计算协方差矩阵  输入应该是m*n,输出为m*m
    eigvalue, eigvector = np.linalg.eig(cov_data)  # 求协方差的特征值和特征向量  m * m
    index = np.argsort(-eigvalue)  # 从大到小排序
    Vec = eigvector[:,index[:k]]  # 取出对应的特征向量K个
    final_data = data.dot(Vec)  # 降维后的数据  n*m  m*k
    return final_data

def plot(data):
    # fig = plt.figure()
    plt.plot(data[:,0],data[:,1],c='red')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# 进行降维
raw_data = np.random.random([100,50])
# print(raw_data.shape)
data_reduce = pca(raw_data,2)
plot(data_reduce)




# a=np.array([1,3,5,7])
# print(a.shape)
# index = np.argsort(-a)
# print(index.shape,index)
# b=np.array([[1,2,3,1],[4,5,6,2],[7,8,9,3]])
# print(b.shape)
# print(index[:2])
#
# Vec = b[:,index[:2]]
# print(Vec,Vec.shape)


