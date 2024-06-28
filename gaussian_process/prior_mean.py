import types
import numpy
from util import value_or_value


class PriorMean:
    """A class describing the prior mean of a Gaussian Process"""

    def evaluate(self, x: numpy.ndarray) -> numpy.ndarray:
        """Evaluates the mean function at x

        Args:
            x (numpy.ndarray): The x value(s) at which to evaluate the prior mean function

        Returns:
            numpy.ndarray: The prior mean function value(s) at x
        """
        assert False, "Not implemented"

    def __call__(self, x: numpy.ndarray) -> numpy.ndarray:
        """See :func:`PriorMean.evaluate`"""
        return self.evaluate(x)

    def derivative(self, x: numpy.ndarray) -> numpy.ndarray:
        """Evaluates the derivative of the mean function at x

        Args:
            x (numpy.ndarray): The x value(s) at which to evaluate the prior mean function derivative

        Returns:
            numpy.ndarray: The prior mean function derivative value(s) at x
        """
        assert False, "Not implemented"


class NumpyPriorMean(PriorMean):
    """A class that offers a numpy member"""

    def __init__(self, np: types.ModuleType) -> None:
        """Initialize the numpy member

        Args:
            np (types.ModuleType): The numpy module to use or None for default numpy
        """
        super().__init__()

        self.np = value_or_value(np, numpy)


class ConstantMean(NumpyPriorMean):
    """The constant prior mean function m(x)=c"""

    def __init__(self, c: float = 0.0, np: types.ModuleType = None) -> None:
        """Initialize the mean function m(x)=c

        Args:
            c (float, optional): The constant parameter of the constant mean. Defaults to 0.0.
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
        """
        super().__init__(np)

        self.c = c

    def evaluate(self, x: numpy.ndarray) -> numpy.ndarray:
        return self.np.full(self.np.shape(x), self.c)

    def derivative(self, x: numpy.ndarray) -> numpy.ndarray:
        return self.np.full(self.np.shape(x), 0.0)


class ZeroMean(ConstantMean):
    """The zero mean function m(x)=0"""

    def __init__(self, np: types.ModuleType = None) -> None:
        """Initialize the mean function m(x)=0

        Args:
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
        """
        super().__init__(0.0, np)


class LinearMean(NumpyPriorMean):
    """The linear mean function m(x)=ax+b"""

    def __init__(
        self, a: float = 1.0, b: float = 0.0, np: types.ModuleType = None
    ) -> None:
        """Initialize the mean function m(x)=ax+b

        Args:
            a (float, optional): The a parameter of the linear function. Defaults to 1.0.
            b (float, optional): The b parameter of the linear function. Defaults to 0.0.
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
        """
        super().__init__(np)

        self.a = a
        self.b = b

    def evaluate(self, x: numpy.ndarray) -> numpy.ndarray:
        return self.a * x + self.b

    def derivative(self, x: numpy.ndarray) -> numpy.ndarray:
        return self.np.full(self.np.shape(x), self.a)


class SquareMean(NumpyPriorMean):
    """The square mean function m(x)=ax^2+bx+c"""

    def __init__(
        self,
        a: float = 1.0,
        b: float = 0.0,
        c: float = 0.0,
        np: types.ModuleType = None,
    ) -> None:
        """Initialize the mean function m(x)=ax^2+bx+c

        Args:
            a (float, optional): The a parameter of the quadratic function. Defaults to 1.0.
            b (float, optional): The b parameter of the quadratic function. Defaults to 0.0.
            c (float, optional): The c paramater of the quadratic function. Defaults to 0.0.
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
        """
        super().__init__(np)

        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, x: numpy.ndarray) -> numpy.ndarray:
        return self.a * x**2 + self.b * x + self.c

    def derivative(self, x: numpy.ndarray) -> numpy.ndarray:
        return 2 * self.a * x + self.b
