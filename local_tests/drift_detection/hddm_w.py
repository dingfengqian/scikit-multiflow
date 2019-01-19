import numpy as np
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class HDDM_W(BaseDriftDetector):
    def __init__(self, a_w=0.005, a_d=0.001):
        super().__init__()

        # Controls how much weight is given to more recent data compared to older data.
        # Smaller values mean less weight given to recent data.
        self.weight = 0.05
        self.total_count = 0  # sample count
        self.X_min = 0  # Moving Average x (1, n)
        self.X_max = 0
        self.Z_ = 0  # Weighted Moving Average x (1, m) m>n   position n is cut point
        self.min_cut_count = 0  # from start to cut point
        self.max_cut_count = 0
        self.n_count = 0  # from start to current
        self.X_min_epsilon = 0
        self.X_max_epsilon = 0
        self.Z_epsilon = 0
        self.Z_D = 0
        self.X_min_D = 0
        self.X_max_D = 0
        self.two_side = True
        self.a_w = a_w
        self.a_d = a_d

    def add_element(self, prediction):
        '''
        a prediction from classifier comes
        Parameters
        ----------
        prediction int (either 0 or 1)
        -------
        '''
        if self.min_cut_count == self.n_count and self.max_cut_count == self.n_count and self.n_count == 0:
            self.Z_D = 1
            self.X_max_D = 1
            self.X_min_D = 1
            self.Z_ = prediction
            self.total_count += 1
            self.n_count += 1
            self.min_cut_count += 1
            self.max_cut_count += 1
        else:
            self.total_count += 1
            self.n_count += 1
            # ------------------ update-----------------------
            # use Theorem 4 to calculate the Dn
            self.Z_D = self.weight * self.weight + (1 - self.weight) * (1 - self.weight) * self.Z_D
            self.Z_ = self.weight * prediction + (1 - self.weight) * self.Z_
        # use Corollary 5 calculate the error bound
        self.Z_epsilon = np.sqrt(self.Z_D * np.log(1 / self.a_d) / 2)

        if self.min_cut_count == self.n_count and self.n_count == 1:
            self.X_min = self.Z_
            self.X_min_epsilon = self.Z_epsilon

        if self.two_side and self.max_cut_count == self.n_count and self.n_count == 1:
            self.X_max = self.Z_
            self.X_max_epsilon = self.Z_epsilon


        if self.Z_ + self.Z_epsilon <= self.X_min + self.X_min_epsilon:
            self.X_min = self.Z_
            self.X_min_D = self.Z_D
            self.X_min_epsilon = self.Z_epsilon
            self.min_cut_count = self.n_count

        if self.two_side and self.Z_ - self.Z_epsilon >= self.X_max - self.X_max_epsilon:
            self.X_max = self.Z_
            self.X_max_D = self.Z_D
            self.X_max_epsilon = self.Z_epsilon
            self.max_cut_count = self.n_count

        # print(self.total_count, 'Z ', self.Z_, ' ', self.Z_epsilon, ' Max error tolerance', self.Z_ + self.Z_epsilon)
        # print(self.total_count, 'X ', self.X_, ' ', self.X_epsilon, 'Max error tolerance', self.X_ + self.X_epsilon)
        # print('Z_D', self.Z_D, ' X_D',self.X_D)

        print(self.total_count, 'Z ', self.Z_, ' ', self.Z_epsilon, ' Max error tolerance', self.Z_ + self.Z_epsilon)
        print(self.total_count, 'X ', self.X_max, ' ', self.X_max_epsilon, 'Max error tolerance', self.X_max + self.X_max_epsilon)
        print('Z_D', self.Z_D, ' X_D',self.X_max_D)

        if not self.two_side:
            if self.meanIncrEstimate(self.min_cut_count, self.n_count, self.X_min, self.X_min_D, self.Z_, self.Z_D,
                                     self.a_d):
                self.in_concept_change = True
                self.in_warning_zone = False
                self.reset()
            elif self.meanIncrEstimate(self.min_cut_count, self.n_count, self.X_min, self.X_min_D, self.Z_, self.Z_D,
                                       self.a_w):
                self.in_concept_change = False
                self.in_warning_zone = True
            else:
                self.in_warning_zone = False
                self.in_concept_change = False
        else:
            if self.meanIncrEstimate(self.min_cut_count, self.n_count, self.X_min, self.X_min_D, self.Z_, self.Z_D, self.a_d):
                self.in_concept_change = True
                self.in_warning_zone = False
                self.reset()
            elif self.meanIncrEstimate(self.min_cut_count, self.n_count, self.X_min, self.X_min_D, self.Z_, self.Z_D, self.a_w):
                self.in_concept_change = False
                self.in_warning_zone = True
            elif self.meanDecrEstimate(self.max_cut_count, self.n_count, self.X_max, self.X_max_D, self.Z_, self.Z_D, self.a_d):
                self.in_concept_change = True
                self.in_warning_zone = False
                self.reset()
            elif self.meanDecrEstimate(self.max_cut_count, self.n_count, self.X_max, self.X_max_D, self.Z_, self.Z_D, self.a_w):
                self.in_concept_change = False
                self.in_warning_zone = True
            else:
                self.in_warning_zone = False
                self.in_concept_change = False

    def meanIncrEstimate(self, cut_count, n_count, X_, X_D, Z_, Z_D, confidence):
        '''
        check whether population mean increment
        use Corollary 5 (6) (7) to calculate the error bound
        Parameters
        ----------
        cut_count
        n_count
        X_
        X_epsilon
        Z_
        Z_epsilon
        confidence

        Returns
        -------

        '''
        if cut_count == n_count:
            return  False
        confidence_epsilon = np.sqrt((X_D + Z_D) * np.log(1 / confidence) / 2)
        print(confidence_epsilon)
        return (Z_ - X_) >= confidence_epsilon

    def meanDecrEstimate(self, cut_count, n_count, X_, X_D, Z_, Z_D, confidence):
        '''
        check whether population mean decrement
        use Corollary 5 (6) (7) to calculate the error bound
        Parameters
        ----------
        cut_count
        n_count
        X_
        X_epsilon
        Z_
        Z_epsilon
        confidence

        Returns
        -------

        '''
        if cut_count == n_count:
            return  False
        confidence_epsilon = np.sqrt((X_D + Z_D) * np.log(1 / confidence) / 2)
        print(confidence_epsilon)
        return (X_ - Z_) >= confidence_epsilon

    def get_info(self):
        pass

    def reset(self):
        self.total_count = 0  # sample count
        self.X_min = 0  # Moving Average x (1, n)
        self.X_max = 0
        self.Z_ = 0  # Weighted Moving Average x (1, m) m>n   position n is cut point
        self.min_cut_count = 0  # from start to cut point
        self.max_cut_count = 0
        self.n_count = 0  # from start to current
        self.X_min_epsilon = 0
        self.X_max_epsilon = 0
        self.Z_epsilon = 0
        self.Z_D = 0
        self.X_min_D = 0
        self.X_max_D = 0
