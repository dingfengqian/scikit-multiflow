import numpy as np
from skmultiflow.drift_detection.ddm import DDM

if __name__ == '__main__':
    ddm = DDM()
    data_stream = np.random.randint(2, size=2000)
    for i in range(999,1500):
        data_stream[i] = 1
    for i in range(2000):
        ddm.add_element(data_stream[i])
        if ddm.detected_warning_zone():
            print('Warning zone has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
        if ddm.detected_change():
            print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))