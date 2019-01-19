import matlab.engine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from arch import arch_model


def calStaArbitrageParam(preDataA, preDataB):
    print(np.shape(preDataA))
    sig = np.max(np.abs(preDataA - preDataB)) / 2
    val = np.zeros_like(preDataA)
    for j in range(len(preDataA)):
        val[j] = fmin(lambda alpha : np.square(sum(np.exp(-np.square(preDataA - preDataA[j]) / 2 / sig / sig)*(preDataB - alpha*preDataA))), 1, disp=False)
    #print(val)
    spread = preDataB - val*preDataA
    msp = np.mean(preDataB - val*preDataA)
    mspread = spread - msp
    garch11 = arch_model(mspread, p=1, q=1)
    res = garch11.fit()
    print(res)
    return spread, mspread

def draw(data, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(data)), data)
    ax.set_title(title, fontdict={'family' : 'SimHei'})
    plt.show()

def draw(data1, data2,title1, title2):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(range(len(data1)), data1)
    ax2 = fig.add_subplot(212)
    ax2.plot(range(len(data2)), data2)
    ax1.set_title(title1, fontdict={'family' : 'SimHei'})
    ax2.set_title(title2, fontdict={'family' : 'SimHei'})
    plt.show()

if __name__ == '__main__':

    df_A = pd.read_csv('./CloseA.csv', header=None).as_matrix().astype(np.double)
    df_B = pd.read_csv('./CloseB.csv', header=None).as_matrix().astype(np.double)

    start = 1
    window = 100
    #result = calStaArbitrageParam(df_A[267:367], df_B[267:367])
    spread, mspread = calStaArbitrageParam(df_A[start:window+start], df_B[start:window+start])
    #draw(mspread, '去中心化残差')
    draw(spread, mspread, '残差', '去中心化残差')
