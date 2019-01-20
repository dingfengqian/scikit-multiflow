#import matlab.engine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from arch import arch_model
from hddm_w import HDDM_W
from sklearn.preprocessing import MinMaxScaler

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

def draw2(data1, data2,title1, title2):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(range(len(data1)), data1)
    ax2 = fig.add_subplot(212)
    ax2.plot(range(len(data2)), data2)
    ax1.set_title(title1, fontdict={'family' : 'SimHei'})
    ax2.set_title(title2, fontdict={'family' : 'SimHei'})
    plt.show()

def showPlot(data, data2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(data)), data)
    ax.plot(range(len(data2)), data2)
    plt.legend(['acc','bound'])
    plt.show()


def hddm_w_tst(spread):
    hddm = HDDM_W()
    data_stream = spread

    average_prediciton = []
    average_prediction_bound = []
    for i in range(len(spread)):
        hddm.add_element(data_stream[i])
        if hddm.detected_warning_zone():
            print('Warning zone has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
        if hddm.detected_change():
            print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
        average_prediciton.append(hddm.Z_)
        average_prediction_bound.append(hddm.Z_epsilon)
        # print('X', hddm.X_, ' ', 'Z', hddm.Z_)
        # print('X_e', hddm.X_epsilon, ' ', 'Z', hddm.Z_epsilon)
    showPlot(average_prediciton, average_prediction_bound)


if __name__ == '__main__':

    df_A = pd.read_csv('./CloseA.csv', header=None).values.astype(np.double)
    df_B = pd.read_csv('./CloseB.csv', header=None).values.astype(np.double)

    start = 1
    window = 300
    #result = calStaArbitrageParam(df_A[267:367], df_B[267:367])
    spread, mspread = calStaArbitrageParam(df_A[start:window+start], df_B[start:window+start])
    #draw(mspread, '去中心化残差')
    #draw2(spread, mspread, '残差', '去中心化残差')
    print(mspread)

    mm = MinMaxScaler()
    result = mm.fit_transform(mspread)
    #draw(result,'归一化残差')
    #draw2(mspread, result, '去中心化残差','归一化去中心化残差')
    print(result)
    hddm_w_tst(spread)
