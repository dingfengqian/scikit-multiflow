import numpy as np
from skmultiflow.drift_detection.ddm import DDM
import matplotlib.pyplot as plt

def showPlot(data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(data)), data)
    plt.legend(['acc'])
    plt.show()

def tst():
    ddm = DDM()
    data_stream = np.random.randint(2, size=2000)
    average_prediciton = []
    for i in range(1000, 1200):
        data_stream[i] = 1
    for i in range(1500, 1700):
        data_stream[i] = 1
    for i in range(2000):
        ddm.add_element(data_stream[i])
        if ddm.detected_warning_zone():
            print('Warning zone has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
        if ddm.detected_change():
            print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
        average_prediciton.append(ddm.miss_prob)

        # print('X', hddm.X_, ' ', 'Z', hddm.Z_)
        # print('X_e', hddm.X_epsilon, ' ', 'Z', hddm.Z_epsilon)
    showPlot(average_prediciton)

def tst2():
    ddm = DDM()
    data_stream = np.random.binomial(1, 0.5, 2000)

    change_data_stream = np.random.binomial(1, 0.8, 500)

    average_prediciton = []
    for i in range(1000, 1200):
        data_stream[i] = change_data_stream[i - 1000]
    for i in range(1500, 1700):
        data_stream[i] = change_data_stream[i + 200 - 1500]
    for i in range(2000):
        ddm.add_element(data_stream[i])
        if ddm.detected_warning_zone():
            print('Warning zone has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
        if ddm.detected_change():
            print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
        average_prediciton.append(ddm.miss_prob)

        # print('X', hddm.X_, ' ', 'Z', hddm.Z_)
        # print('X_e', hddm.X_epsilon, ' ', 'Z', hddm.Z_epsilon)
    showPlot(average_prediciton)

if __name__ == '__main__':
    tst2()