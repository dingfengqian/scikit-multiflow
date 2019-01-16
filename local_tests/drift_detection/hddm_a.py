import numpy as np
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class HDDM_A(BaseDriftDetector):
    def __init__(self, a_w=0.005, a_d=0.001):
        super().__init__()
        self.total_count = 0           # sample count
        self.X_ = 0                    # Moving Average x (1, n)
        self.Z_ = 0                    # Moving Average x (1, m) m>n   position n is cut point
        self.cut_count = 0             # from start to cut point
        self.n_count = 0               # from start to current
        self.X_epsilon = 0
        self.Z_epsilon = 0
        self.a_w = a_w
        self.a_d = a_d
        self.reset()

    def add_element(self, prediction):
        '''
        a prediction from classifier comes
        Parameters
        ----------
        prediction int (either 0 or 1)
        -------
        '''
        self.total_count += 1
        self.n_count += 1
        if self.cut_count == 0:
            self.cut_count = self.total_count
        if self.n_count == 0:
            self.n_count = self.total_count
# ------------------ update ---------------------
        # use Theorem 1 calculate error bounds
        self.Z_epsilon = np.sqrt(1.0 / (2 * self.n_count) * np.log(1.0 / self.a_d))
        self.Z_ = self.Z_ + (prediction - self.Z_) / self.n_count
        if self.cut_count == self.n_count and self.n_count == 1:
            self.X_epsilon = self.Z_epsilon
            self.X_ = self.Z_

        # cut point update
        if self.Z_ + self.Z_epsilon <= self.X_ + self.X_epsilon:
            self.X_ = self.Z_
            self.X_epsilon = self.Z_epsilon
            self.cut_count = self.n_count

        print(self.total_count, 'Z ', self.Z_, ' ', self.Z_epsilon, ' Max error tolerance', self.Z_ + self.Z_epsilon)
        print(self.total_count, 'X ', self.X_, ' ', self.X_epsilon, 'Max error tolerance', self.X_ + self.X_epsilon)

        # check whether drift arise
        if self.meanIncrEstimate(self.X_, self.Z_, self.cut_count, self.n_count, self.a_d):
            self.in_concept_change = True
            self.in_warning_zone = False
            self.reset()
            # init
        elif self.meanIncrEstimate(self.X_, self.Z_, self.cut_count, self.n_count, self.a_w):
            self.in_concept_change = False
            self.in_warning_zone = True
        else:
            self.in_concept_change = False
            self.in_warning_zone = False

    def meanIncrEstimate(self, X_, Z_, X_count, Z_count, confidence):
        '''
        check population mean increment by Corollary 9 to calculate the error bound of E[Z] - E[X]
        Parameters
        ----------
        X_
        Z_
        X_count
        Z_count
        confidence

        Returns
        -------
        True or False
        '''
        if X_count == Z_count:
            return False
        m = Z_count - X_count
        n = X_count
        confidence_epsilon = np.sqrt(m / (2 * n * (n + m)) * np.log(1 / confidence))
        return (Z_ - X_) >= confidence_epsilon


    def reset(self):
        super().reset()
        self.total_count = 0  # sample count
        self.X_ = 0  # Moving Average x (1, n)
        self.Z_ = 0  # Moving Average x (1, m) m>n   position n is cut point
        self.cut_count = 0  # from start to cut point
        self.n_count = 0  # from start to current
        self.X_epsilon = 0
        self.Z_epsilon = 0


    def get_info(self):
        pass


