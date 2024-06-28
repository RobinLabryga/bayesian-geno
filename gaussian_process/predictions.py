import numpy
from gaussian_process import GaussianProcess


# TODO: Comments on GPPrediction members
class GPPrediction:
    """A class that saves intermediate computations performed by the Gaussian Process to predict at a specific x."""

    def __init__(self, x: numpy.ndarray, gp: GaussianProcess) -> None:
        self.x = x
        self.gp = gp
        self.__K_x_x: numpy.ndarray = None
        self.__K_x_x_derivative_lhs: numpy.ndarray = None
        self.__K_x_x_derivative_rhs: numpy.ndarray = None
        self.__K_x_x_derivative_lhsrhs: numpy.ndarray = None
        self.__K_x_known: numpy.ndarray = None
        self.__K_x_known_derivative_lhs: numpy.ndarray = None
        self.__K_known_x: numpy.ndarray = None
        self.__K_known_x_derivative_rhs: numpy.ndarray = None
        self.__mean: numpy.ndarray = None
        self.__covariance: numpy.ndarray = None
        self.__variance: numpy.ndarray = None
        self.__std_deviation: numpy.ndarray = None
        self.__derivative_mean: numpy.ndarray = None
        self.__derivative_covariance: numpy.ndarray = None
        self.__derivative_variance: numpy.ndarray = None
        self.__derivative_std_deviation: numpy.ndarray = None
        self.__std_deviation_derivative: numpy.ndarray = None

    @property
    def mean(self) -> numpy.ndarray:
        if self.__mean is None:
            self.__mean = self.gp.mean(self.x, K_x_known=self._K_x_known)
        return self.__mean

    @property
    def covariance(self) -> numpy.ndarray:
        if self.__covariance is None:
            self.__covariance = self.gp.covariance(
                self.x,
                K_x_x=self._K_x_x,
                K_x_known=self._K_x_known,
                K_known_x=self._K_known_x,
            )
        return self.__covariance

    @property
    def variance(self) -> numpy.ndarray:
        if self.__variance is None:
            self.__variance = self.gp.variance(self.x, covariance=self.covariance)
        return self.__variance

    @property
    def std_deviation(self) -> numpy.ndarray:
        if self.__std_deviation is None:
            self.__std_deviation = self.gp.std_deviation(self.x, variance=self.variance)
        return self.__std_deviation

    @property
    def derivative_mean(self) -> numpy.ndarray:
        if self.__derivative_mean is None:
            self.__derivative_mean = self.gp.derivative_mean(
                self.x, K_x_known_derivative_lhs=self._K_x_known_derivative_lhs
            )
        return self.__derivative_mean

    @property
    def derivative_covariance(self) -> numpy.ndarray:
        if self.__derivative_covariance is None:
            self.__derivative_covariance = self.gp.derivative_covariance(
                self.x,
                K_x_x_derivative_lhs=self._K_x_x_derivative_lhs,
                K_x_known=self._K_x_known,
                K_known_x=self._K_known_x,
                K_x_known_derivative_lhs=self._K_x_known_derivative_lhs,
                K_known_x_derivative_rhs=self._K_known_x_derivative_rhs,
            )
        return self.__derivative_covariance

    @property
    def derivative_variance(self) -> numpy.ndarray:
        if self.__derivative_variance is None:
            self.__derivative_variance = self.gp.derivative_variance(
                self.x, derivative_covariance=self.derivative_covariance
            )
        return self.__derivative_variance

    @property
    def derivative_std_deviation(self) -> numpy.ndarray:
        if self.__derivative_std_deviation is None:
            self.__derivative_std_deviation = self.gp.derivative_std_deviation(
                self.x, derivative_variance=self.derivative_variance
            )
        return self.__derivative_std_deviation

    @property
    def std_deviation_derivative(self) -> numpy.ndarray:
        if self.__std_deviation_derivative is None:
            self.__std_deviation_derivative = self.gp.std_deviation_derivative(
                self.x,
                derivative_variance=self.derivative_variance,
                std_deviation=self.std_deviation,
            )
        return self.__std_deviation_derivative

    @property
    def _K_x_x(self):
        if self.__K_x_x is None:
            self.__K_x_x = self.gp._compute_K_x_x(self.x)
        return self.__K_x_x

    @property
    def _K_x_x_derivative_lhs(self):
        if self.__K_x_x_derivative_lhs is None:
            self.__K_x_x_derivative_lhs = self.gp._compute_K_x_x_derivative_lhs(self.x)
        return self.__K_x_x_derivative_lhs

    @property
    def _K_x_x_derivative_rhs(self):
        if self.__K_x_x_derivative_rhs is None:
            self.__K_x_x_derivative_rhs = self.gp._compute_K_x_x_derivative_rhs(self.x)
        return self._K_x_x_derivative_rhs

    @property
    def _K_x_x_derivative_lhsrhs(self):
        if self.__K_x_x_derivative_lhsrhs is None:
            self.__K_x_x_derivative_lhsrhs = self.gp._compute_K_x_x_derivative_lhsrhs(
                self.x
            )
        return self._K_x_x_derivative_lhsrhs

    @property
    def _K_x_known(self):
        if self.__K_x_known is None:
            self.__K_x_known = self.gp._compute_K_x_known(self.x)
        return self.__K_x_known

    @property
    def _K_x_known_derivative_lhs(self):
        if self.__K_x_known_derivative_lhs is None:
            self.__K_x_known_derivative_lhs = self.gp._compute_K_x_known_derivative_lhs(
                self.x
            )
        return self.__K_x_known_derivative_lhs

    @property
    def _K_known_x(self):
        if self.__K_known_x is None and not self.gp.kernel.is_symmetric:
            self.__K_known_x = self.gp._compute_K_known_x(self.x)
        return self.__K_known_x

    def _init_K_known_x(self):
        pass

    @property
    def _K_known_x_derivative_rhs(self):
        if self.__K_known_x_derivative_rhs is None and not self.gp.kernel.is_symmetric:
            self.__K_known_x_derivative_rhs = self.gp._compute_K_known_x_derivative_rhs(
                self.x
            )
        return self.__K_known_x_derivative_rhs
