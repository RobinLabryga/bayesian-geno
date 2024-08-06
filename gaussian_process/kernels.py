# TODO: Look more into mean square differentiable

import numpy
import types
from util import value_or_value

class Kernel:
    """A function that maps two values into the reals"""

    @property
    def is_positive_definite(self) -> bool:
        """True if the Kernel is positive definite, False otherwise"""
        return False

    @property
    def is_symmetric(self) -> bool:
        """True if k(a,b)=k(b,a), False otherwise"""
        return False

    @property
    def is_covariance_function(self) -> bool:
        """True if the Kernel is symmetric and positive definite, False otherwise"""
        return self.is_symmetric and self.is_positive_definite

    def evaluate(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        """Evaluates the kernel at k(a,b)

        Args:
            a (numpy.ndarray): The lhs argument
            b (numpy.ndarray): The rhs argument

        Returns:
            numpy.ndarray: The result of k(a,b)
        """
        assert False, "Not implemented"

    def __call__(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        """See :func:`Kernel.evaluate`"""
        return self.evaluate(a, b)

    def derivative_lhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        """Evaluates the derivative with respect to the lhs argument of the kernel

        Args:
            a (numpy.ndarray): The lhs argument
            b (numpy.ndarray): The rhs argument

        Returns:
            numpy.ndarray: The derivative of the kernel with respect to the lhs argument
        """
        assert False, "Not implemented"

    def derivative_rhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        """Evaluates the derivative with respect to the rhs argument of the kernel

        Args:
            a (numpy.ndarray): The lhs argument
            b (numpy.ndarray): The rhs argument

        Returns:
            numpy.ndarray: The derivative of the kernel with respect to the rhs argument
        """
        assert False, "Not implemented"

    def derivative_lhsrhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        """Evaluates the derivative with respect to the lhs and rhs argument of the kernel

        Args:
            a (numpy.ndarray): The lhs argument
            b (numpy.ndarray): The rhs argument

        Returns:
            numpy.ndarray: The derivative of the kernel with respect to the lhs and rhs argument
        """
        assert False, "Not implemented"


class PositiveDefiniteKernel(Kernel):
    def __init__(self) -> None:
        super().__init__()

    @Kernel.is_positive_definite.getter
    def is_positive_definite(self):
        return True


class SymmetricKernel(Kernel):
    def __init__(self) -> None:
        super().__init__()

    @Kernel.is_symmetric.getter
    def is_symmetric(self):
        return True


class CovarianceFunction(PositiveDefiniteKernel, SymmetricKernel):
    def __init__(self) -> None:
        super().__init__()

class KernelSum(Kernel):
    """A kernel that is the sum of two other kernels

    If both underlying kernels are covariance functions, the sum is also a covariance function.
    """

    def __init__(self, lhs: Kernel, rhs: Kernel) -> None:
        """Initializes a Kernel that is the result of summing two kernels

        If the lhs and rhs kernels are covariance functions, the resulting sum is also a covariance function.

        Args:
            lhs (Kernel): The lhs kernel of the sum
            rhs (Kernel): The rhs kernel of the sum
        """
        self.lhs = lhs
        self.rhs = rhs

    @Kernel.is_positive_definite.getter
    def is_positive_definite(self):
        return self.lhs.is_positive_definite and self.rhs.is_positive_definite

    @Kernel.is_symmetric.getter
    def is_symmetric(self):
        return self.lhs.is_symmetric and self.rhs.is_symmetric

    def evaluate(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return self.lhs.evaluate(a, b) + self.rhs.evaluate(a, b)

    def derivative_lhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return self.lhs.derivative_lhs(a, b) + self.rhs.derivative_lhs(a, b)

    def derivative_rhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return self.lhs.derivative_rhs(a, b) + self.rhs.derivative_rhs(a, b)

    def derivative_lhsrhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return self.lhs.derivative_lhsrhs(a, b) + self.rhs.derivative_lhsrhs(a, b)


class KernelProduct(Kernel):
    """A kernel that is the product of two other kernels

    If both underlying kernels are covariance functions, the product is also a covariance function
    """

    def __init__(self, lhs: Kernel, rhs: Kernel) -> None:
        """Initializes a Kernel that is the result of the product of two kernels

        If the lhs and rhs kernels are covariance functions, the resulting product is also a covariance function.

        Args:
            lhs (Kernel): The lhs kernel of the product
            rhs (Kernel): The rhs kernel of the product
        """
        self.lhs = lhs
        self.rhs = rhs

    @Kernel.is_positive_definite.getter
    def is_positive_definite(self):
        return self.lhs.is_positive_definite and self.rhs.is_positive_definite

    @Kernel.is_symmetric.getter
    def is_symmetric(self):
        return self.lhs.is_symmetric and self.rhs.is_symmetric

    def evaluate(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return self.lhs.evaluate(a, b) * self.rhs.evaluate(a, b)

    def derivative_lhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        f_lhs = self.lhs.evaluate(a, b)
        ga_lhs = self.lhs.derivative_lhs(a, b)
        f_rhs = self.rhs.evaluate(a, b)
        ga_rhs = self.rhs.derivative_lhs(a, b)

        return f_lhs * ga_rhs + ga_lhs * f_rhs

    def derivative_rhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        f_lhs = self.lhs.evaluate(a, b)
        gb_lhs = self.lhs.derivative_rhs(a, b)
        f_rhs = self.rhs.evaluate(a, b)
        gb_rhs = self.rhs.derivative_rhs(a, b)

        return f_lhs * gb_rhs + gb_lhs * f_rhs

    def derivative_lhsrhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        f_lhs = self.lhs.evaluate(a, b)
        ga_lhs = self.lhs.derivative_lhs(a, b)
        gb_lhs = self.lhs.derivative_rhs(a, b)
        gab_lhs = self.lhs.derivative_lhsrhs(a, b)
        f_rhs = self.rhs.evaluate(a, b)
        ga_rhs = self.rhs.derivative_lhs(a, b)
        gb_rhs = self.rhs.derivative_rhs(a, b)
        gab_rhs = self.rhs.derivative_lhsrhs(a, b)

        return f_lhs * gab_rhs + ga_lhs * gb_rhs + gb_lhs * ga_rhs + gab_lhs * f_rhs


class SquaredExponentialKernel(CovarianceFunction):
    """Equivalent to Matern_{infty}

    Often also referred to as the Gaussian Kernel or the RBF (RBF is technically a class of Kernels).
    """

    def __init__(self, l: float = 1.0, np: types.ModuleType = None) -> None:
        """Initializes the squared exponential kernel

        Args:
            l (float, optional): The length scale parameter of the kernel. Defaults to 1.0.
            np (types.ModuleType, optional): The numpy module to use. None for the default numpy. Defaults to None.
        """
        super().__init__()
        self.np = value_or_value(np, numpy)

        self.l = l

    def evaluate(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return self.np.exp(-(1 / (2 * self.l**2)) * self.np.abs(a - b) ** 2)

    def derivative_lhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        d = a - b
        r = self.np.abs(d)
        t0 = self.l**2
        t1 = self.np.exp(-(r**2 / (2 * t0)))
        s = self.np.sign(d)
        s = self.np.where(s == 0, 1, s)

        return -r * s * t1 / t0

    def derivative_rhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return -self.derivative_lhs(a, b)

    def derivative_lhsrhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        d = a - b
        r = self.np.abs(d)
        t0 = self.l**2
        t1 = r**2
        t2 = self.np.exp(-(t1 / (2 * t0)))

        return t2 / t0 * (1 - t1 / t0)


class ExponentialKernel(CovarianceFunction):
    """Equivalent to gamma-exponential kernel with gamma=1 and Matern_{frac{1}{2}}"""

    def __init__(self, l: float = 1.0, np: types.ModuleType = None) -> None:
        """Initialize the Exponential Kernel

        Args:
            l (float, optional): The length scale parameter of the kernel. Defaults to 1.0.
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
        """
        super().__init__()
        self.np = value_or_value(np, numpy)

        self.l = l

    def evaluate(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return self.np.exp(-(self.np.abs(a - b) / (self.l)))

    def derivative_lhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        d = a - b
        r = self.np.abs(d)
        s = self.np.sign(d)
        ex = self.np.exp(-(r / self.l))
        return -s * ex / self.l

    def derivative_rhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return -self.derivative_lhs(a, b)

    def derivative_lhsrhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        r = self.np.abs(a - b)
        ex = self.np.exp(-(r / self.l))
        return -ex / self.l**2


class GammaExponentialKernel(CovarianceFunction):
    """The gamma-exponential kernel

    Is a covariance function.
    """

    def __init__(
        self, gamma=float, l: float = 1.0, np: types.ModuleType = None
    ) -> None:
        """Initialize the Gamma exponential kernel.

        Args:
            gamma (_type_, optional): The exponent (must be in (0.0, 2.0]). Defaults to float.
            l (float, optional): The length scale parameter. Defaults to 1.0.
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
        """
        super().__init__()
        self.np = value_or_value(np, numpy)

        self.gamma = gamma
        assert 0 < self.gamma and self.gamma <= 2.0
        self.l = l

    def evaluate(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        r = self.np.abs(a - b)
        return self.np.exp(-((r / self.l) ** self.gamma))

    def derivative_lhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        d = a - b
        r = self.np.abs(d)
        s = self.np.sign(d)
        ex = self.np.exp(-((r / self.l) ** self.gamma))
        return -self.gamma / self.l**self.gamma * (r ** (self.gamma - 1)) * s * ex

    def derivative_rhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return -self.derivative_lhs(a, b)

    def derivative_lhsrhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        l = self.l
        g = self.gamma
        d = a - b
        r = self.np.abs(d)
        ex = self.np.exp(-((r / l) ** g))
        t0 = l**-g
        return -g * t0 * ex * (g * (r ** (2 * g - 2)) * t0 - (r ** (g - 2)) * (g - 1))


class CubicKernel(SymmetricKernel):
    """Equivalent to Polyharmonic Spline with k=3."""

    def __init__(self, np: types.ModuleType = None) -> None:
        """Initialize the cubic kernel

        Args:
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
        """
        super().__init__()
        self.np = value_or_value(np, numpy)

    def evaluate(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        r = self.np.abs(a - b)
        return r**3

    def derivative_lhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        r = self.np.abs(a - b)
        s = self.np.sign(a - b)
        return 3 * r**2 * s

    def derivative_rhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return -self.derivative_lhs(a, b)

    def derivative_lhsrhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        r = self.np.abs(a - b)
        return -6 * r


class PolyharmonicSplineKernel(SymmetricKernel):
    """The Polyharmonic Spline Kernel"""

    def __init__(self, k: int, np: types.ModuleType = None) -> None:
        """Initialize the polyharmonic spline kernel

        Args:
            k (int): The polynomial
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
        """
        super().__init__()
        self.np = value_or_value(np, numpy)

        self.k = k
        assert k > 0

    def evaluate(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        r = self.np.abs(a - b)
        if self.k % 2 == 0:
            # return r**k * ln(r)
            return r ** (self.k - 1) * self.np.log(r**r)  # To avoid ln(0)
        else:
            return r**self.k

    def derivative_lhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        d = a - b
        r = self.np.abs(d)
        s = self.np.where(self.np.sign(d) == 0, 1, self.np.sign(d))
        if self.k % 2 == 0:
            # sign(a-b)(kr**(k-1)ln(r)+r**k/r) = sign(a-b)(kr**(k-1)ln(r)+r**(k-1))
            return (
                self.k * r ** (self.k - 2) * self.np.log(r**r) + r ** (self.k - 1)
            ) * s
        else:
            return self.k * r ** (self.k - 1) * s

    def derivative_rhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return -self.derivative_lhs(a, b)

    def derivative_lhsrhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        if self.k == 1:
            return self.np.full((self.np.broadcast(a, b).shape), 0.0)

        r = self.np.abs(a - b)
        if self.k % 2 == 0:
            # k(k-1)r**(k-2)ln(r) + kr**(k-1)/r + (k-1)r**(k-2) = k-1)r**(k-2)ln(r) + (k+(k-1))r**(k-2)
            t0 = self.k * (self.k - 1) * r ** (self.k - 3) * self.np.log(r**r)
            t1 = (2 * self.k - 1) * r ** (self.k - 2)
            return -(t0 + t1)
        else:
            return -self.k * (self.k - 1) * r ** (self.k - 2)


class RationalQuadraticKernel(CovarianceFunction):
    """The Rational Quadratic Kernel"""

    def __init__(
        self, alpha: float = 1.0, l: float = 1.0, np: types.ModuleType = None
    ) -> None:
        """Initialize the Rational Quadratic Kernel

        Args:
            alpha (float, optional): The exponent (must be >= 0). Defaults to 1.0.
            l (float, optional): The length scale parameter. Defaults to 1.0.
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
        """
        super().__init__()
        self.np = value_or_value(np, numpy)

        self.alpha = alpha
        assert self.alpha >= 0.0
        self.l = l

    def evaluate(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        r = self.np.abs(a - b)
        return (1 + r**2 / (2 * self.alpha * self.l**2)) ** (-self.alpha)

    def derivative_lhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        g = self.alpha
        l = self.l
        d = a - b
        r = self.np.abs(d)
        s = self.np.sign(d)
        t0 = l**2
        t1 = 1 + ((r**2) / (2 * g * t0))
        return -s * (((t1 ** -(1 + g)) * r) / t0)

    def derivative_rhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return -self.derivative_lhs(a, b)

    def derivative_lhsrhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        np = self.np
        g = self.alpha
        l = self.l
        d = a - b
        r = np.abs(d)
        t0 = l**2
        t1 = r**2
        t2 = 1 + (t1 / (2 * g * t0))
        t3 = 1 + g
        t4 = t2**-t3
        return -(((t2 ** -(2 + g) * t1 * t3) / (l**4 * g)) - (t4 / t0))


class Matern1_5Kernel(CovarianceFunction):
    """Matern_{frac{3}{2}}"""

    def __init__(self, l: float = 1.0, np: types.ModuleType = None) -> None:
        """Initialize the Matern (3/2) kernel

        Args:
            l (float, optional): The length scale parameter. Defaults to 1.0.
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
        """
        super().__init__()
        self.np = value_or_value(np, numpy)

        self.l = l

    def evaluate(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        r = self.np.abs(a - b)
        t0 = (self.np.sqrt(3) * r) / (self.l)
        return (1 + t0) * self.np.exp(-t0)

    def derivative_lhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        t0 = a - b
        r = self.np.abs(t0)
        s = self.np.sign(t0)
        t5 = self.np.exp(-(self.np.sqrt(3) * r) / self.l)
        t6 = (3 * r * s) / self.l**2
        return -t6 * t5

    def derivative_rhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return -self.derivative_lhs(a, b)

    def derivative_lhsrhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        t0 = a - b
        r = self.np.abs(t0)
        t2 = (self.np.sqrt(3) * r) / self.l
        return (3 / self.l**2 * (1 - t2)) * self.np.exp(-t2)


class Matern2_5Kernel(CovarianceFunction):
    """Matern_{frac{5}{2}}"""

    def __init__(self, l: float = 1.0, np: types.ModuleType = None) -> None:
        """Initialize the Matern (5/2) kernel

        Args:
            l (float, optional): The length scale parameter. Defaults to 1.0.
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
        """
        super().__init__()
        self.np = value_or_value(np, numpy)

        self.l = l

    def evaluate(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        r = self.np.abs(a - b)
        t0 = (self.np.sqrt(5) * r) / (self.l)
        t1 = (5 * r**2) / (3 * self.l**2)
        return (1 + t0 + t1) * self.np.exp(-t0)

    def derivative_lhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        sr5 = self.np.sqrt(5)
        l = self.l
        d = a - b
        r = self.np.abs(d)
        s = self.np.sign(d)
        t0 = (sr5 * r) / l
        ex = self.np.exp(-t0)

        return -5 / (3 * l**2) * s * r * ex * (1 + t0)

    def derivative_rhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return -self.derivative_lhs(a, b)

    def derivative_lhsrhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        l = self.l
        sr5 = self.np.sqrt(5)
        d = a - b
        r = self.np.abs(d)
        t0 = sr5 * r / l
        ex = self.np.exp(-t0)
        return 5 / (3 * l**2) * ex * (1 + t0 - t0**2)


class ConstantKernel(CovarianceFunction):
    """Constant Kernel"""

    def __init__(self, c: float = 0.0, np: types.ModuleType = None) -> None:
        """Initialize the constant kernel

        Args:
            c (float, optional): The constant the kernel evaluates to. Defaults to 0.0.
            np (types.ModuleType, optional): The numpy module in use. None for default numpy. Defaults to None.
        """
        super().__init__()
        self.np = value_or_value(np, numpy)

        self.c = c
        assert c >= 0.0

    def evaluate(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return self.np.full(self.np.broadcast(a, b).shape, self.c)

    def derivative_lhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return self.np.full(self.np.broadcast(a, b).shape, 0.0)

    def derivative_rhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return self.np.full(self.np.broadcast(a, b).shape, 0.0)

    def derivative_lhsrhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return self.np.full(self.np.broadcast(a, b).shape, 0.0)


class LinearKernel(CovarianceFunction):
    """Linear Kernel"""

    def __init__(self, a: float = 1.0, np: types.ModuleType = None) -> None:
        """Initialize the linear kernel

        Args:
            a (float, optional): The slope parameter. Defaults to 1.0.
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
        """
        super().__init__()
        self.np = value_or_value(np, numpy)

        assert a >= 0.0
        self.a = a

    def evaluate(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        # We should use a.T * b, but since we are in 1D it does not matter and is easier for meshgrid
        return self.a * (a * b)

    def derivative_lhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return self.a * b

    def derivative_rhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return self.a * a

    def derivative_lhsrhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return self.np.full(self.np.broadcast(a, b).shape, self.a)


class PolynomialKernel(CovarianceFunction):
    """Polynomial Kernel"""

    def __init__(self, p: int, sigma: float = 1.0, np: types.ModuleType = None) -> None:
        """Initialize the polynomial kernel

        Args:
            p (int): The polynomial degree
            sigma (float, optional): The offset of the base. Defaults to 1.0.
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
        """
        super().__init__()
        self.np = value_or_value(np, numpy)

        self.p = p
        assert self.p > 0
        self.sigma = sigma

    def evaluate(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        # We should use a.T * b, but since we are in 1D it does not matter and is easier for meshgrid
        return (a * b + self.sigma) ** self.p

    def derivative_lhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return b * self.p * (a * b + self.sigma) ** (self.p - 1)

    def derivative_rhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return a * self.p * (a * b + self.sigma) ** (self.p - 1)

    def derivative_lhsrhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        t0 = a * b
        t1 = t0 + self.sigma
        return self.p * (t1 ** (self.p - 1) + t0 * (self.p - 1) * t1 ** (self.p - 2))


class BrownianKernel(SymmetricKernel):
    """Brownian Kernel"""

    def __init__(self, np: types.ModuleType = None) -> None:
        """Initialize the Brownian kernel

        Args:
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
        """
        super().__init__()
        self.np = value_or_value(np, numpy)

    def evaluate(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return self.np.minimum(a, b)

    def derivative_lhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return self.np.where(a < b, 1.0, 0.0)

    def derivative_rhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return self.np.where(a > b, 1.0, 0.0)

    def derivative_lhsrhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return self.np.full(self.np.broadcast(a, b).shape, 0.0)


class SincKernel(SymmetricKernel):
    """Sinc Kernel

    Is not a covariance function.
    """

    def __init__(self, np: types.ModuleType = None) -> None:
        """Initialize the sinc kernel

        Args:
            np (types.ModuleType, optional): The numpy module to use. None for default numpy. Defaults to None.
        """
        super().__init__()
        self.np = value_or_value(np, numpy)

    def evaluate(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        r = self.np.abs(a - b)
        return self.np.where(r == 0.0, 1.0, self.np.sin(r) / r)

    def derivative_lhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        d = a - b
        r = self.np.abs(d)
        s = self.np.sign(d)
        g = s * (self.np.cos(r) / r - self.np.sin(r) / r**2)
        return self.np.where(r == 0.0, 0.0, g)

    def derivative_rhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        return -self.derivative_lhs(a, b)

    def derivative_lhsrhs(self, a: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
        r = self.np.abs(a - b)
        t_2 = self.np.sin(r)
        T_7 = self.np.cos(r) / r**2
        g = 2 * T_7 + (t_2 / r) * (1 - 2 / (r**2))
        return self.np.where(r == 0.0, 1.0 / 3.0, g)
