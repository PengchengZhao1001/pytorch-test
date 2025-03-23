import matplotlib.pyplot as plt
import numpy as np

def dataset():
    x = np.array([15, 20, 25, 30, 35, 40])
    y = np.array([330, 345, 365, 405, 430, 450])
    return x,y

def regression(x,y):
    x_means=np.mean(x)
    y_means=np.mean(y)
    S_xx=np.sum((x-x_means)**2)
    S_xy=np.sum((x-x_means)*(y-y_means))
    beta=S_xy/S_xx
    b=y_means-beta*x_means
    y_regress = b + beta * x

    a=np.sum((y-y_regress)**2)
    print(a,x.shape[0])
    c=(a/(x.shape[0]-2))**0.5
    print(c)
    T=abs(beta/(c/(S_xx**0.5)))
    print(T)
    print(S_xx)

    return beta,b,y_regress


if __name__ == '__main__':
    x,y=dataset()
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'ro')
    beta, b, y_regress= regression(x, y)


    plt.figure(1)
    plt.subplot(1, 2, 2)
    plt.plot(x, y_regress,'b',x, y, 'ro')
    plt.show()
    print("beta=[%f],b=[%f]" % (beta,b))





