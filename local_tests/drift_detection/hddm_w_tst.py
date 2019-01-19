import numpy as np
import matplotlib.pyplot as plt
from hddm_w import HDDM_W

def showPlot(data, data2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(data)), data)
    ax.plot(range(len(data2)), data2)
    plt.legend(['acc','bound'])
    plt.show()

def tst():
    hddm = HDDM_W()
    data_stream = np.random.randint(2, size=2000)

    average_prediciton = []
    average_prediction_bound = []
    for i in range(1000, 1200):
        data_stream[i] = 1
    for i in range(1500, 1700):
        data_stream[i] = 1
    for i in range(2000):
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

def tst2():
    hddm = HDDM_W()
    data_stream = np.random.binomial(1, 0.9, 2000)

    change_data_stream = np.random.binomial(1, 0.5, 500)

    average_prediciton = []
    average_prediction_bound = []
    for i in range(1000, 1250):
        data_stream[i] = change_data_stream[i - 1000]
    for i in range(1500, 1750):
        data_stream[i] = change_data_stream[i + 200 - 1500]
    for i in range(2000):
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

    tst2()

