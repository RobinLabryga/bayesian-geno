from gaussian_process import GaussianProcess, GPPrediction
import numpy
import types
import scipy.stats
from util import value_or_value


class AcquisitionFunction:
    def __init__(self, gp: GaussianProcess) -> None:
        """Initialize the underlying Gaussian Process

        Args:
            gp (GaussianProcess): The underlying Gaussian Process
        """
        self.gp = gp

    def evaluate(self, x: numpy.ndarray) -> numpy.ndarray:
        """Evaluate the acquisition at x

        Args:
            x (numpy.ndarray): The value(s) to evaluate the acquisition at

        Returns:
            numpy.ndarray: The acquisition at x
        """
        assert False, "Not implemented"

    def __call__(self, x: numpy.ndarray) -> numpy.ndarray:
        """See :func:`AcquisitionFunction.evaluate`"""
        return self.evaluate(x)

    def derivative(self, x: numpy.ndarray) -> numpy.ndarray:
        """Evaluates the derivative of the acquisition function at x

        Args:
            x (numpy.ndarray): The value(s) to evaluate the acquisition derivative at.

        Returns:
            numpy.ndarray: The acquisition derivative at x
        """
        assert False, "Not implemented"

    def value_derivative(self, x: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Evaluates function value and derivative of the acquisition function at x

        Args:
            x (numpy.ndarray): The value(s) to evaluate at

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: The acquisition value and derivative
        """
        return self.evaluate(x), self.derivative(x)


class NumpyAcquisitionFunction(AcquisitionFunction):
    """A class that provides a numpy module member to derived classes"""

    def __init__(self, gp: GaussianProcess, np: types.ModuleType | None) -> None:
        """Initializes the numpy member

        Args:
            gp (GaussianProcess): The Gaussian process over which to calculate the acquisition
            np (types.ModuleType | None): The numpy module to use
        """
        super().__init__(gp)

        self.np = value_or_value(np, numpy)


class LowerConfidenceBound(NumpyAcquisitionFunction):
    def __init__(
        self, gp: GaussianProcess, lcb_factor: float = 2.0, np: types.ModuleType = None
    ) -> None:
        """Initialize the acquisition function -(m(x) - lcb_factor * sigma(x))

        Args:
            gp (GaussianProcess): The underlying Gaussian Process
            lcb_factor (float, optional): The factor on the standard deviation. Defaults to 2.0.
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
        """
        super().__init__(gp, np)

        self.lcb_factor = lcb_factor

    def evaluate(self, x: numpy.ndarray) -> numpy.ndarray:
        pred = GPPrediction(x, self.gp)

        return -pred.mean + self.lcb_factor * self.np.abs(pred.std_deviation)

    def derivative(self, x: numpy.ndarray) -> numpy.ndarray:
        pred = GPPrediction(x, self.gp)

        return -pred.derivative_mean + self.lcb_factor * self.np.abs(
            pred.std_deviation_derivative
        ) * self.np.sign(pred.std_deviation_derivative)

    def value_derivative(self, x: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        pred = GPPrediction(x, self.gp)

        value = -pred.mean + self.lcb_factor * self.np.abs(pred.std_deviation)
        derivative = -pred.derivative_mean + self.lcb_factor * self.np.abs(
            pred.std_deviation_derivative
        ) * self.np.sign(pred.std_deviation_derivative)

        return value, derivative


class UpperConfidenceBound(NumpyAcquisitionFunction):
    def __init__(
        self, gp: GaussianProcess, ucb_factor: float = 2.0, np: types.ModuleType = None
    ) -> None:
        """Initialize the acquisition function m(x) + ucb_factor * sigma(x)

        Args:
            gp (GaussianProcess): The underlying Gaussian Process
            ucb_factor (float, optional): The factor on the standard deviation. Defaults to 2.0.
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
        """
        super().__init__(gp, np)

        self.ucb_factor = ucb_factor

    def evaluate(self, x: numpy.ndarray) -> numpy.ndarray:
        pred = GPPrediction(x, self.gp)

        return pred.mean + self.ucb_factor * self.np.abs(pred.std_deviation)

    def derivative(self, x: numpy.ndarray) -> numpy.ndarray:
        pred = GPPrediction(x, self.gp)
        return pred.derivative_mean + self.ucb_factor * self.np.abs(
            pred.std_deviation_derivative
        ) * self.np.sign(pred.std_deviation_derivative)

    def value_derivative(self, x: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        pred = GPPrediction(x, self.gp)

        value = pred.mean + self.ucb_factor * self.np.abs(pred.std_deviation)
        derivative = pred.derivative_mean + self.ucb_factor * self.np.abs(
            pred.std_deviation_derivative
        ) * self.np.sign(pred.std_deviation_derivative)

        return value, derivative


class LowerConfidenceBoundVariance(NumpyAcquisitionFunction):
    def __init__(
        self, gp: GaussianProcess, lcb_factor: float = 2.0, np: types.ModuleType = None
    ) -> None:
        """Initialize the acquisition function -(m(x) - lcb_factor * sigma^2(x))

        Args:
            gp (GaussianProcess): The underlying Gaussian Process
            lcb_factor (float, optional): The factor on the variance. Defaults to 2.0.
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
        """
        super().__init__(gp, np)

        self.lcb_factor = lcb_factor

    def evaluate(self, x: numpy.ndarray) -> numpy.ndarray:
        mean, variance = self.gp(x)
        return -mean + self.lcb_factor * self.np.abs(variance)

    def derivative(self, x: numpy.ndarray) -> numpy.ndarray:
        pred = GPPrediction(x, self.gp)
        return -pred.derivative_mean + self.lcb_factor * self.np.abs(
            pred.derivative_variance
        ) * self.np.sign(pred.derivative_variance)

    def value_derivative(self, x: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        pred = GPPrediction(x, self.gp)

        value = -pred.mean + self.lcb_factor * self.np.abs(pred.variance)
        derivative = -pred.derivative_mean + self.lcb_factor * self.np.abs(
            pred.derivative_variance
        ) * self.np.sign(pred.derivative_variance)

        return value, derivative


class UpperConfidenceBoundVariance(NumpyAcquisitionFunction):
    def __init__(
        self, gp: GaussianProcess, ucb_factor: float = 2.0, np: types.ModuleType = None
    ) -> None:
        """Initialize the acquisition function m(x) + ucb_factor * sigma^2(x)

        Args:
            gp (GaussianProcess): The underlying Gaussian Process
            ucb_factor (float, optional): The factor on the variance. Defaults to 2.0.
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
        """
        super().__init__(gp, np)

        self.ucb_factor = ucb_factor

    def evaluate(self, x: numpy.ndarray) -> numpy.ndarray:
        mean, variance = self.gp(x)
        return mean + self.ucb_factor * self.np.abs(variance)

    def derivative(self, x: numpy.ndarray) -> numpy.ndarray:
        pred = GPPrediction(x, self.gp)

        return pred.derivative_mean + self.ucb_factor * self.np.abs(
            pred.derivative_variance
        ) * self.np.sign(pred.derivative_variance)

    def value_derivative(self, x: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        pred = GPPrediction(x, self.gp)

        value = pred.mean + self.ucb_factor * self.np.abs(pred.variance)
        derivative = pred.derivative_mean + self.ucb_factor * self.np.abs(
            pred.derivative_variance
        ) * self.np.sign(pred.derivative_variance)

        return value, derivative


class ProbabilityOfImprovement_maximization(NumpyAcquisitionFunction):
    def __init__(
        self,
        gp: GaussianProcess,
        f_best,
        tradeoff: float = 0.05,
        np: types.ModuleType = None,
        norm: types.ModuleType = None,
    ) -> None:
        """Initialize the acquisition function

        Args:
            gp (GaussianProcess): The underlying Gaussian Process
            f_best (_type_): The best known function values so far.
            tradeoff (float, optional): The tradeoff parameter. Defaults to 0.05.
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
            norm (types.ModuleType, optional): The norm module to use. None for scipy.stats.norm. Defaults to None.
        """
        super().__init__(gp, np)

        self.f_best = f_best
        self.tradeoff = tradeoff

        self.norm = value_or_value(norm, scipy.stats.norm)

    def evaluate(self, x: numpy.ndarray) -> numpy.ndarray:
        pred = GPPrediction(x, self.gp)

        # TODO: Make multi-x compatible
        if pred.std_deviation == 0.0:
            return pred.std_deviation
        return self.norm.cdf(
            (pred.mean - (self.f_best + self.tradeoff)) / pred.std_deviation
        )

    def derivative(self, x: numpy.ndarray) -> numpy.ndarray:
        pred = GPPrediction(x, self.gp)

        if pred.std_deviation == 0.0:
            return pred.std_deviation

        t0 = pred.mean - (self.f_best + self.tradeoff)
        Z = t0 / pred.std_deviation
        dZ = (
            pred.derivative_mean / pred.std_deviation
            + t0 * pred.std_deviation_derivative / pred.variance
        )
        return dZ * self.norm.pdf(Z)

    def value_derivative(self, x: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        pred = GPPrediction(x, self.gp)

        if pred.std_deviation == 0.0:
            return pred.std_deviation, pred.std_deviation

        t0 = pred.mean - (self.f_best + self.tradeoff)
        Z = t0 / pred.std_deviation
        dZ = (
            pred.derivative_mean / pred.std_deviation
            + t0 * pred.std_deviation_derivative / pred.variance
        )

        value = self.norm.cdf(Z)
        derivative = dZ * self.norm.pdf(Z)

        return value, derivative


class ProbabilityOfImprovement_minimization(NumpyAcquisitionFunction):
    def __init__(
        self,
        gp: GaussianProcess,
        f_best,
        tradeoff: float = 0.05,
        np: types.ModuleType = None,
        norm: types.ModuleType = None,
    ) -> None:
        """Initialize the acquisition function

        Args:
            gp (GaussianProcess): The underlying Gaussian Process
            f_best (_type_): The best known function values so far.
            tradeoff (float, optional): The tradeoff parameter. Defaults to 0.05.
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
            norm (types.ModuleType, optional): The norm module to use. None for scipy.stats.norm. Defaults to None.
        """
        super().__init__(gp, np)

        self.f_best = f_best
        self.tradeoff = tradeoff

        self.norm = value_or_value(norm, scipy.stats.norm)

    def evaluate(self, x: numpy.ndarray) -> numpy.ndarray:
        pred = GPPrediction(x, self.gp)

        if pred.std_deviation == 0.0:
            return pred.std_deviation

        return self.norm.cdf(
            ((self.f_best - self.tradeoff) - pred.mean) / pred.std_deviation
        )

    def derivative(self, x: numpy.ndarray) -> numpy.ndarray:
        pred = GPPrediction(x, self.gp)

        if pred.std_deviation == 0.0:
            return pred.std_deviation

        t0 = (self.f_best - self.tradeoff) - pred.mean
        Z = t0 / pred.std_deviation
        dZ = (
            -pred.derivative_mean / pred.std_deviation
            + t0 * pred.std_deviation_derivative / pred.variance
        )

        return dZ * self.norm.pdf(Z)

    def value_derivative(self, x: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        pred = GPPrediction(x, self.gp)

        if pred.std_deviation == 0.0:
            return pred.std_deviation, pred.std_deviation

        t0 = (self.f_best - self.tradeoff) - pred.mean
        Z = t0 / pred.std_deviation
        dZ = (
            -pred.derivative_mean / pred.std_deviation
            + t0 * pred.std_deviation_derivative / pred.variance
        )

        value = self.norm.cdf(Z)
        derivative = dZ * self.norm.pdf(Z)

        return value, derivative


class ExpectedImprovement_maximization(NumpyAcquisitionFunction):
    def __init__(
        self,
        gp: GaussianProcess,
        f_best,
        tradeoff: float = 0.01,
        np: types.ModuleType = None,
        norm: types.ModuleType = None,
    ) -> None:
        """Initialize the acquisition function

        Args:
            gp (GaussianProcess): The underlying Gaussian Process
            f_best (_type_): The best known function values so far.
            tradeoff (float, optional): The tradeoff parameter. Defaults to 0.05.
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
            norm (types.ModuleType, optional): The norm module to use. None for scipy.stats.norm. Defaults to None.
        """
        super().__init__(gp, np)

        self.f_best = f_best
        self.tradeoff = tradeoff

        self.norm = value_or_value(norm, scipy.stats.norm)

    def evaluate(self, x: numpy.ndarray) -> numpy.ndarray:
        pred = GPPrediction(x, self.gp)

        if pred.std_deviation == 0.0:
            return pred.std_deviation
        t0 = pred.mean - (self.f_best + self.tradeoff)
        Z = t0 / pred.std_deviation
        return t0 * self.norm.cdf(Z) + pred.std_deviation * self.norm.pdf(Z)

    def derivative(self, x: numpy.ndarray) -> numpy.ndarray:
        pred = GPPrediction(x, self.gp)

        if pred.std_deviation == 0.0:
            return pred.std_deviation

        t0 = pred.mean - (self.f_best + self.tradeoff)
        Z = t0 / pred.std_deviation
        dZ = (
            pred.derivative_mean / pred.std_deviation
            + t0 * pred.std_deviation_derivative / pred.variance
        )
        pdf = self.norm.pdf(Z)
        dcdf = dZ * pdf
        t1 = pred.derivative_mean * self.norm.cdf(Z)
        t2 = t0 * dcdf
        t3 = pred.std_deviation_derivative
        t4 = pred.std_deviation * Z * dcdf
        return t1 + t2 + t3 - t4

    def value_derivative(self, x: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        pred = GPPrediction(x, self.gp)

        if pred.std_deviation == 0.0:
            return pred.std_deviation, pred.std_deviation

        t0 = pred.mean - (self.f_best + self.tradeoff)
        Z = t0 / pred.std_deviation
        dZ = (
            pred.derivative_mean / pred.std_deviation
            + t0 * pred.std_deviation_derivative / pred.variance
        )
        pdf = self.norm.pdf(Z)
        cdf = self.norm.cdf(Z)
        dcdf = dZ * pdf
        t1 = pred.derivative_mean * cdf
        t2 = t0 * dcdf
        t3 = pred.std_deviation_derivative
        t4 = pred.std_deviation * Z * dcdf

        value = t0 * cdf + pred.std_deviation * pdf
        derivative = t1 + t2 + t3 - t4

        return value, derivative


class ExpectedImprovement_minimization(NumpyAcquisitionFunction):
    def __init__(
        self,
        gp: GaussianProcess,
        f_best,
        tradeoff: float = 0.01,
        np: types.ModuleType = None,
        norm: types.ModuleType = None,
    ) -> None:
        """Initialize the acquisition function

        Args:
            gp (GaussianProcess): The underlying Gaussian Process
            f_best (_type_): The best known function values so far.
            tradeoff (float, optional): The tradeoff parameter. Defaults to 0.05.
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
            norm (types.ModuleType, optional): The norm module to use. None for scipy.stats.norm. Defaults to None.
        """
        super().__init__(gp, np)

        self.f_best = f_best
        self.tradeoff = tradeoff

        self.norm = value_or_value(norm, scipy.stats.norm)

    def evaluate(self, x: numpy.ndarray) -> numpy.ndarray:
        pred = GPPrediction(x, self.gp)

        if pred.std_deviation == 0.0:
            return pred.std_deviation

        t0 = (self.f_best - self.tradeoff) - pred.mean
        Z = t0 / pred.std_deviation
        return t0 * self.norm.cdf(Z) + pred.std_deviation * self.norm.pdf(Z)

    def derivative(self, x: numpy.ndarray) -> numpy.ndarray:
        pred = GPPrediction(x, self.gp)

        if pred.std_deviation == 0.0:
            return pred.std_deviation

        t0 = (self.f_best - self.tradeoff) - pred.mean
        Z = t0 / pred.std_deviation
        dZ = (
            -pred.derivative_mean / pred.std_deviation
            + t0 * pred.std_deviation_derivative / pred.variance
        )
        pdf = self.norm.pdf(Z)
        dcdf = dZ * pdf
        t1 = pred.derivative_mean * self.norm.cdf(Z)
        t2 = t0 * dcdf
        t3 = pred.std_deviation_derivative
        t4 = pred.std_deviation * Z * dcdf
        return -t1 + t2 + t3 - t4

    def value_derivative(self, x: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        pred = GPPrediction(x, self.gp)

        if pred.std_deviation == 0.0:
            return pred.std_deviation, pred.std_deviation

        t0 = (self.f_best - self.tradeoff) - pred.mean
        Z = t0 / pred.std_deviation
        dZ = (
            -pred.derivative_mean / pred.std_deviation
            + t0 * pred.std_deviation_derivative / pred.variance
        )
        cdf = self.norm.cdf(Z)
        pdf = self.norm.pdf(Z)
        dcdf = dZ * pdf
        t1 = pred.derivative_mean * self.norm.cdf(Z)
        t2 = t0 * dcdf
        t3 = pred.std_deviation_derivative
        t4 = pred.std_deviation * Z * dcdf

        value = t0 * cdf + pred.std_deviation * pdf
        derivative = -t1 + t2 + t3 - t4

        return value, derivative


class GP_UCB(NumpyAcquisitionFunction):
    def __init__(
        self,
        gp: GaussianProcess,
        t: int,
        dim: int,
        nu: float = 1.0,
        delta: float = 0.5,
        np: types.ModuleType = None,
    ) -> None:
        """Initialize the acquisition function

        Args:
            gp (GaussianProcess): The underlying Gaussian Process.
            t (int): The number of iterations so far.
            dim (int): The dimensionality of the problem.
            nu (float, optional): The nu parameter. Defaults to 1.0.
            delta (float, optional): The delta parameter. Defaults to 0.5.
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
        """
        super().__init__(gp, np)

        taut = 2.0 * self.np.log(
            (t + 1) ** ((dim / 2.0) + 2.0) * (self.np.pi**2 / (3.0 * delta))
        )
        self.gpucb_factor = self.np.sqrt(nu * taut)

    def evaluate(self, x: numpy.ndarray) -> numpy.ndarray:
        pred = GPPrediction(x, self.gp)

        return pred.mean + self.gpucb_factor * self.np.abs(pred.std_deviation)

    def derivative(self, x: numpy.ndarray) -> numpy.ndarray:
        pred = GPPrediction(x, self.gp)

        return pred.derivative_mean + self.gpucb_factor * self.np.abs(
            pred.std_deviation_derivative
        ) * self.np.sign(pred.std_deviation_derivative)

    def value_derivative(self, x: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        pred = GPPrediction(x, self.gp)

        value = pred.mean + self.gpucb_factor * self.np.abs(pred.std_deviation)
        derivative = pred.derivative_mean + self.gpucb_factor * self.np.abs(
            pred.std_deviation_derivative
        ) * self.np.sign(pred.std_deviation_derivative)

        return value, derivative


class GP_LCB(NumpyAcquisitionFunction):
    def __init__(
        self,
        gp: GaussianProcess,
        t: int,
        dim: int,
        nu: float = 1.0,
        delta: float = 0.5,
        np: types.ModuleType = None,
    ) -> None:
        """Initialize the acquisition function

        Args:
            gp (GaussianProcess): The underlying Gaussian Process.
            t (int): The number of iterations so far.
            dim (int): The dimensionality of the problem.
            nu (float, optional): The nu parameter. Defaults to 1.0.
            delta (float, optional): The delta parameter. Defaults to 0.5.
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
        """
        super().__init__(gp, np)

        taut = 2.0 * self.np.log(
            (t + 1) ** ((dim / 2.0) + 2.0) * (self.np.pi**2 / (3.0 * delta))
        )
        self.gplcb_factor = self.np.sqrt(nu * taut)

    def evaluate(self, x: numpy.ndarray) -> numpy.ndarray:
        pred = GPPrediction(x, self.gp)

        return -pred.mean + self.gplcb_factor * self.np.abs(pred.std_deviation)

    def derivative(self, x: numpy.ndarray) -> numpy.ndarray:
        pred = GPPrediction(x, self.gp)

        return -pred.derivative_mean + self.gplcb_factor * self.np.abs(
            pred.std_deviation_derivative
        ) * self.np.sign(pred.std_deviation_derivative)

    def value_derivative(self, x: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        pred = GPPrediction(x, self.gp)

        value = -pred.mean + self.gplcb_factor * self.np.abs(pred.std_deviation)
        derivative = -pred.derivative_mean + self.gplcb_factor * self.np.abs(
            pred.std_deviation_derivative
        ) * self.np.sign(pred.std_deviation_derivative)

        return value, derivative


class GP_UCB_Variance(NumpyAcquisitionFunction):
    def __init__(
        self,
        gp: GaussianProcess,
        t: int,
        dim: int,
        nu: float = 1.0,
        delta: float = 0.5,
        np: types.ModuleType = None,
    ) -> None:
        """Initialize the acquisition function

        Args:
            gp (GaussianProcess): The underlying Gaussian Process.
            t (int): The number of iterations so far.
            dim (int): The dimensionality of the problem.
            nu (float, optional): The nu parameter. Defaults to 1.0.
            delta (float, optional): The delta parameter. Defaults to 0.5.
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
        """
        super().__init__(gp, np)

        taut = 2.0 * self.np.log(
            (t + 1) ** ((dim / 2.0) + 2.0) * (self.np.pi**2 / (3.0 * delta))
        )
        self.gpucb_factor = self.np.sqrt(nu * taut)

    def evaluate(self, x: numpy.ndarray) -> numpy.ndarray:
        mean, variance = self.gp(x)
        return mean + self.gpucb_factor * self.np.abs(variance)

    def derivative(self, x: numpy.ndarray) -> numpy.ndarray:
        pred = GPPrediction(x, self.gp)

        return pred.derivative_mean + self.gpucb_factor * self.np.abs(
            pred.derivative_variance
        ) * self.np.sign(pred.derivative_variance)

    def value_derivative(self, x: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        pred = GPPrediction(x, self.gp)

        value = pred.mean + self.gpucb_factor * self.np.abs(pred.variance)
        derivative = pred.derivative_mean + self.gpucb_factor * self.np.abs(
            pred.derivative_variance
        ) * self.np.sign(pred.derivative_variance)

        return value, derivative


class GP_LCB_Variance(NumpyAcquisitionFunction):
    def __init__(
        self,
        gp: GaussianProcess,
        t: int,
        dim: int,
        nu: float = 1.0,
        delta: float = 0.5,
        np: types.ModuleType = None,
    ) -> None:
        """Initialize the acquisition function

        Args:
            gp (GaussianProcess): The underlying Gaussian Process.
            t (int): The number of iterations so far.
            dim (int): The dimensionality of the problem.
            nu (float, optional): The nu parameter. Defaults to 1.0.
            delta (float, optional): The delta parameter. Defaults to 0.5.
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
        """
        super().__init__(gp, np)

        taut = 2.0 * self.np.log(
            (t + 1) ** ((dim / 2.0) + 2.0) * (self.np.pi**2 / (3.0 * delta))
        )
        self.gplcb_factor = self.np.sqrt(nu * taut)

    def evaluate(self, x: numpy.ndarray) -> numpy.ndarray:
        mean, variance = self.gp(x)
        return -mean + self.gplcb_factor * self.np.abs(variance)

    def derivative(self, x: numpy.ndarray) -> numpy.ndarray:
        pred = GPPrediction(x, self.gp)

        return -pred.derivative_mean + self.gplcb_factor * self.np.abs(
            pred.derivative_variance
        ) * self.np.sign(pred.derivative_variance)

    def value_derivative(self, x: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        pred = GPPrediction(x, self.gp)

        value = -pred.mean + self.gplcb_factor * self.np.abs(pred.variance)
        derivative = -pred.derivative_mean + self.gplcb_factor * self.np.abs(
            pred.derivative_variance
        ) * self.np.sign(pred.derivative_variance)

        return value, derivative
