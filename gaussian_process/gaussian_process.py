"""Gaussian Processes"""

from types import ModuleType
from typing import Callable
from gaussian_process.kernels import Kernel
from gaussian_process.prior_mean import PriorMean, ZeroMean
import scipy.linalg
import numpy
from util import value_or_func, value_or_value


# TODO: Comments on GP members
# TODO: Sort members to have public interface at the top and everything else at the bottom


class GaussianProcess:
    """A Gaussian Process

    The kernel/covariance function can be specified
    Is able to be conditioned on function values, gradients
    Can incorporate noise of function values and gradients
    The posterior mean can be specified
    """

    def _compute_gram_matrix(
        self,
        a: numpy.ndarray,
        b: numpy.ndarray,
        kernel: Kernel | Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray] = None,
    ) -> numpy.ndarray:
        if a is None or b is None:
            return None

        kernel = value_or_value(kernel, self.kernel)
        meshgrid = self.np.meshgrid(
            a, b, indexing="ij", sparse=True
        )  # Covariance matrix is matrix of k(x_i, x_j) for all x_i,x_j in X
        return kernel(meshgrid[0], meshgrid[1])

    def _compute_K_known_known(self):
        if self.x_known is None:
            return None

        if self.g_known is not None:
            # K_known_known is a 2 * len(x_known) x 2 * len(x_known) matrix.
            # Top left is len(x_known) x len(x_known) via covarianceFunction
            # Top right is len(x_known) x len(x_known) via covarianceFunctionRHSDerived
            # Bottom left is len(x_known) x len(x_known) via covarianceFunctionLHSDerived
            # Bottom right is len(x_known) x len(x_known) via covarianceFunctionLHSRHSDerived

            covarianceMatrixLHSDerived = self._compute_gram_matrix(
                self.x_known,
                self.x_known,
                kernel=self.kernel.derivative_lhs,
            )
            # Since covarianceFunctionRHSDerived and covarianceFunctionLHSDerived should produce a covariance matrix that is the transpose of the other, we can use the transpose to give a higher chance of avoiding numerical errors.
            covarianceMatrixRHSDerived = (
                covarianceMatrixLHSDerived.T
                if self.kernel.is_symmetric
                else self._compute_gram_matrix(
                    self.x_known,
                    self.x_known,
                    kernel=self.kernel.derivative_rhs,
                )
            )
            covarianceMatrixDefault = self._compute_gram_matrix(
                self.x_known, self.x_known, kernel=self.kernel
            )
            covarianceMatrixLHSRHSDerived = self._compute_gram_matrix(
                self.x_known,
                self.x_known,
                kernel=self.kernel.derivative_lhsrhs,
            )

            # Add noise to covariance matrix along main diagonal
            if self.f_noise is not None:
                covarianceMatrixDefault += self.np.diag(self.f_noise)
            if self.g_noise is not None:
                covarianceMatrixLHSRHSDerived += self.np.diag(self.g_noise)

            # Assemble complete covariance matrix
            return self.np.concatenate(
                (
                    self.np.concatenate(
                        (covarianceMatrixDefault, covarianceMatrixRHSDerived),
                        axis=1,
                    ),
                    self.np.concatenate(
                        (covarianceMatrixLHSDerived, covarianceMatrixLHSRHSDerived),
                        axis=1,
                    ),
                ),
                axis=0,
            )

        K_known_known = self._compute_gram_matrix(self.x_known, self.x_known)

        # Add noise to covariance function
        if self.f_noise is not None:
            K_known_known += self.np.diag(self.f_noise)

        return K_known_known

    def _compute_K_x_known(self, x):
        return (
            self._compute_gram_matrix(x, self.x_known)
            if self.g_known is None
            else (
                # K_x_known is len(X) x 2 * len(x_known) matrix.
                # len(x_known) columns of function values x function values
                # len(x_known) columns of function values x gradient values
                # For first few columns we need covarianceFunction, for last few we need covarianceFunctionRHSDerived
                self.np.concatenate(
                    (
                        self._compute_gram_matrix(x, self.x_known, kernel=self.kernel),
                        self._compute_gram_matrix(
                            x, self.x_known, kernel=self.kernel.derivative_rhs
                        ),
                    ),
                    axis=1,
                )
            )
        )

    def _compute_K_known_x(self, x):
        return (
            self._compute_gram_matrix(self.x_known, x)
            if self.g_known is None
            else (
                # K_x_known is 2 * len(x_known) x len(X) matrix.
                # len(x_known) rows of function values x function values
                # len(x_known) row of gradient values x function values
                # For first few rows we need covarianceFunction, for last few we need covarianceFunctionLHSDerived
                self.np.concatenate(
                    (
                        self._compute_gram_matrix(self.x_known, x, kernel=self.kernel),
                        self._compute_gram_matrix(
                            self.x_known, x, kernel=self.kernel.derivative_lhs
                        ),
                    ),
                    axis=0,
                )
            )
        )

    def _compute_K_x_known_derivative_lhs(self, x):
        return (
            self._compute_gram_matrix(x, self.x_known, self.kernel.derivative_lhs)
            if self.g_known is None
            else self.np.concatenate(
                (
                    self._compute_gram_matrix(
                        x, self.x_known, self.kernel.derivative_lhs
                    ),
                    self._compute_gram_matrix(
                        x, self.x_known, self.kernel.derivative_lhsrhs
                    ),
                ),
                axis=1,
            )
        )

    def _compute_K_known_x_derivative_rhs(self, x):
        return (
            self._compute_gram_matrix(self.x_known, x, self.kernel.derivative_rhs)
            if self.g_known is None
            else self.np.concatenate(
                (
                    self._compute_gram_matrix(
                        self.x_known, x, self.kernel.derivative_rhs
                    ),
                    self._compute_gram_matrix(
                        self.x_known, x, self.kernel.derivative_lhsrhs
                    ),
                ),
                axis=0,
            )
        )

    def _compute_K_x_x(self, x):
        return self._compute_gram_matrix(x, x)

    def _compute_K_x_x_derivative_lhs(self, x):
        return self._compute_gram_matrix(x, x, self.kernel.derivative_lhs)

    def _compute_K_x_x_derivative_rhs(self, x):
        return self._compute_gram_matrix(x, x, self.kernel.derivative_rhs)

    def _compute_K_x_x_derivative_lhsrhs(self, x):
        return self._compute_gram_matrix(x, x, self.kernel.derivative_lhsrhs)

    def _compute_condition_values(self):
        if self.x_known is None:
            return None

        if self.g_known is not None:
            return self.np.concatenate(
                (
                    self.f_known - self.prior_mean(self.x_known),
                    self.g_known - self.prior_mean.derivative(self.x_known),
                ),
                axis=0,
            )

        return self.f_known - self.prior_mean(self.x_known)

    def _compute_K_known_known_inv(self):
        if self.K_known_known is None:
            return None

        return (
            self.np.linalg.inv(self.K_known_known)
            if not self.kernel.is_covariance_function
            else None
        )

    def _compute_K_known_known_cholesky(self):
        if self.K_known_known is None:
            return None

        # We don't catch possible LinAlgErrors here, since the noise is user defined. It might be worth printing a hint, that more noise may solve the issue.
        return (
            scipy.linalg.cholesky(self.K_known_known, lower=True)
            if self.kernel.is_covariance_function
            else None
        )

    def _compute_RW06_2_1_alpha(self):
        if self.K_known_known is None:
            return None

        # TODO: Maybe add skip_finite=True to solve_triangular
        return (
            scipy.linalg.solve_triangular(
                self.K_known_known_cholesky.T,
                scipy.linalg.solve_triangular(
                    self.K_known_known_cholesky,
                    self.condition_values,
                    lower=True,
                ),
                lower=False,
            )
            if self.kernel.is_covariance_function
            else None
        )

    def __init__(
        self,
        kernel: Kernel,
        x_known: numpy.ndarray = None,
        f_known: numpy.ndarray = None,
        g_known: numpy.ndarray = None,
        f_noise: float | numpy.ndarray = None,
        g_noise: float | numpy.ndarray = None,
        prior_mean: PriorMean = None,
        np: ModuleType = None,
        verbose: bool = False,
    ):
        """Initialize the Gaussian Process

        Args:
            kernel (Kernel, optional): The kernel describing the relation between data points. Should but does not have to be a covariance function (Non-positive-definite kernels may still give a result, that are not a Gaussian Process).
            x_known (ndarray, optional): The known x-positions. Defaults to None.
            f_known (ndarray, optional): The function values at the known x-positions. Defaults to None.
            g_known (float | ndarray, optional): The gradient values at the known x-positions. Defaults to None.
            f_noise (float | ndarray, optional): The noise values of the function values at the known x-positions. Defaults to None.
            g_noise (ndarray, optional): The noise values of the gradient values at the known x-positions. Defaults to None.
            prior_mean (PriorMean, optional): A function describing the mean of the Gaussian Process Prior. Defaults to None.
            np (module, optional): The numpy module to use. Defaults to None.
            verbose (bool, optional): True, if the actions should be verbosely printed, False otherwise. Defaults to False.
        """

        assert (
            x_known is None
            and f_known is None
            or x_known is not None
            and f_known is not None
        ), "x_known and f_known should either both be None or neither"
        assert x_known is None or len(x_known) == len(
            f_known
        ), "If x_known and f_known are not None they should both have the same number of elements"
        assert kernel is not None

        assert (
            f_noise is None or type(f_noise) is float or len(f_noise) == len(f_known)
        ), "noise must be a factor or the same length as the corresponding values"
        assert (
            g_noise is None or type(g_noise) is float or len(g_noise) == len(g_known)
        ), "noise must be a factor or the same length as the corresponding values"

        self.verbose = verbose
        """bool: True if verbose, False otherwise"""

        self.np = value_or_value(np, numpy)
        """ModuleType: The numpy in use"""

        self.kernel = kernel
        """Kernel: The Kernel/Covariance function describing the relation between two points"""

        self.prior_mean = value_or_func(prior_mean, ZeroMean)
        """PriorMean: Prior mean of the Gaussian Process"""

        assert x_known is None or len(self.np.unique(x_known)) == len(
            x_known
        ), "Known x values may not contain duplicates"  # TODO: Maybe remove because nlogn (Though solving is slower...)

        self.x_known = x_known
        """ndarray: The known x values"""
        self.f_known = f_known
        """ndarray: The known function values at x"""
        self.g_known = g_known
        """ndarray: The known gradient values at x"""

        self.f_noise = (
            self.np.full(len(x_known), f_noise)
            if type(f_noise) is float
            else f_noise  # ndarray or None
        )
        """The noise of the known function values"""
        self.g_noise = (
            self.np.full(len(x_known), g_noise)
            if type(g_noise) is float
            else g_noise  # ndarray or None
        )
        """The noise of the known gradients"""

        if self.verbose:
            print("Gaussian Process created from points:")
            if self.x_known is not None:
                print(f"x={self.np.array2string(self.x_known, separator=', ')}")
            if self.f_known is not None:
                print(f"f={self.np.array2string(self.f_known, separator=', ')}")
            if self.f_noise is not None:
                print(f"f_noise={self.np.array2string(self.f_noise, separator=', ')}")
            if self.g_known is not None:
                print(f"g={self.np.array2string(self.g_known, separator=', ')}")
            if self.g_noise is not None:
                print(f"g_noise={self.np.array2string(self.g_noise, separator=', ')}")

        self.K_known_known = self._compute_K_known_known()
        """The Gram matrix of the known xs"""

        self.condition_values = self._compute_condition_values()
        """The values used to condition the GP"""

        self.K_known_known_inv = self._compute_K_known_known_inv()
        """ndarray: The inverse of K_known_known"""

        self.K_known_known_cholesky = self._compute_K_known_known_cholesky()
        """ndarray: The cholesky decomposition of K_known_known"""

        self.RW06_2_1_alpha = self._compute_RW06_2_1_alpha()
        """ndarray: The alpha component of RW06 Algorithm 2.1"""

    def _RW06_2_1(
        self, prior_mean: numpy.ndarray, K_x_x: numpy.ndarray, K_known_x: numpy.ndarray
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Algorithm 2.1 from RW06."""

        assert self.K_known_known_cholesky is not None
        assert self.RW06_2_1_alpha is not None

        mean = prior_mean + K_known_x.T @ self.RW06_2_1_alpha
        v = scipy.linalg.solve_triangular(
            self.K_known_known_cholesky, K_known_x, lower=True
        )
        variance = K_x_x - v.T @ v

        return mean, variance

    def _RW06_2_1_modified(
        self,
        prior_mean: numpy.ndarray,
        K_x_x: numpy.ndarray,
        K_x_known: numpy.ndarray,
        K_known_x: numpy.ndarray,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        r"""Algorithm 2.1 from RW06 with modification to use K(X',X) and K(X,X').

        Idea behind algorithm:
        The mean of the Gaussian Process is calculated as

        $
            \mu = K(X',X) @ K(X,X)^{-1} y
        $

        with $X'$ being the test values and $X$ being the known values.
        In the case of a Gaussian process, $K(X,X)$ is positive definite. Thus, it exists cholesky decomposition $K(X,X)=LL^\top$.

        $
            \mu = K(X',X) K(X,X)^{-1} y
            = K(X',X) (LL^\top)^{-1} y
            = K(X',X) \alpha
        $

        We attain $\alpha$ by solving

        $
            \alpha = (LL^\top)^{-1} y
            \Rightarrow LL^\top \alpha = y
            \Rightarrow L^\top \alpha = L \backslash y
            \Rightarrow \alpha = L^\top \backslash (L \backslash y)
        $

        Similarly for the variance of the Gaussian process we have

        $
            \sigma^2=K(X',X') - K(X',X) K(X,X)^{-1} K(X,X')
            = \sigma^2=K(X',X') - K(X',X) (LL^\top)^{-1} K(X,X')
            = \sigma^2=K(X',X') - K(X',X) L^{-\top} L^{-1} K(X,X')
            = \sigma^2=K(X',X') - (L^{-1} K(X',X)^\top)^\top L^{-1} K(X,X')
            = \sigma^2=K(X',X') - (L^{-1} K(X',X)^\top)^\top L^{-1} K(X,X')
            =\sigma^2=K(X',X') - v_l^\top v_r
        $

        with $v_l = L^{-1} K(X',X)^\top$ and $v_r = L^{-1} K(X,X')$
        We attain $v_l$ by solving

        $
            v_l = L^{-1} K(X',X)^\top
            \Rightarrow  L v_l = K(X',X)^\top
            \Rightarrow  v_l = L \backslash K(X',X)^\top
        $

        and $v_r$ by solving

        $
            v_r = L^{-1} K(X,X')
            \Rightarrow L v_r = K(X,X')
            \Rightarrow v_r = L \backslash K(X,X')
        $

        Note that $L$ is a lower triangular matrix.
        """

        assert self.K_known_known_cholesky is not None
        assert self.RW06_2_1_alpha is not None

        mean = prior_mean + K_x_known @ self.RW06_2_1_alpha
        v_l = scipy.linalg.solve_triangular(
            self.K_known_known_cholesky, K_x_known.T, lower=True
        )
        v_r = scipy.linalg.solve_triangular(
            self.K_known_known_cholesky, K_known_x, lower=True
        )
        variance = K_x_x - v_l.T @ v_r

        return mean, variance

    def _evaluate_default(
        self,
        x: numpy.ndarray,
        K_x_x: numpy.ndarray,
        K_x_known: numpy.ndarray,
        K_known_x: numpy.ndarray,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Evaluate the mean and variance of the Gaussian Process at x

        Uses the standard evaluation formula for Gaussian Process posteriors

        Args:
            x (ndarray): The point(s) at which to evaluate the mean and covariance matrix

        Returns:
            (ndarray, ndarray): mean and covariance matrix at x
        """
        return self._mean_default(x, K_x_known=K_x_known), self.np.diag(
            self._covariance_default(
                K_x_x=K_x_x, K_x_known=K_x_known, K_known_x=K_known_x
            )
        )

    def _evaluate_symmetric_kernel(
        self,
        x: numpy.ndarray,
        K_x_x: numpy.ndarray,
        K_x_known: numpy.ndarray,
    ):
        return self._mean_default(x, K_x_known=K_x_known), self.np.diag(
            self._covariance_symmetric_kernel(K_x_x=K_x_x, K_x_known=K_x_known)
        )

    def _evaluate__covariance_kernel(
        self, x: numpy.ndarray, K_x_x: numpy.ndarray, K_x_known: numpy.ndarray
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Evaluate the mean and variance of the Gaussian Process at x

        Uses cholesky decomposition to avoid calculating the inverse of K_known_known offering a more stable evaluation

        Args:
            x (ndarray): The point(s) at which to evaluate the mean and covariance matrix

        Returns:
            (ndarray, ndarray): mean and covariance matrix at x
        """
        return self._mean__covariance_kernel(x, K_x_known=K_x_known), self.np.diag(
            self._covariance__covariance_kernel(K_x_x=K_x_x, K_x_known=K_x_known)
        )

    def evaluate(
        self,
        x: numpy.ndarray,
        K_x_x: numpy.ndarray = None,
        K_x_known: numpy.ndarray = None,
        K_known_x: numpy.ndarray = None,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Evaluate the mean and variance of the Gaussian Process at x

        Uses the default evaluation, if self is not a valid Gaussian Process and more stable evaluation via cholesky decomposition otherwise.

        Args:
            x (ndarray): The point(s) at which to evaluate the mean and covariance matrix

        Returns:
            (ndarray, ndarray): mean and variance at x
        """
        K_x_x = value_or_func(K_x_x, self._compute_K_x_x, x)
        K_x_known = value_or_func(K_x_known, self._compute_K_x_known, x)
        if self.kernel.is_covariance_function:
            return self._evaluate__covariance_kernel(
                x, K_x_x=K_x_x, K_x_known=K_x_known
            )
        if self.kernel.is_symmetric:
            return self._evaluate_symmetric_kernel(x, K_x_x=K_x_x, K_x_known=K_x_known)
        return self._evaluate_default(
            x,
            K_x_x=K_x_x,
            K_x_known=K_x_known,
            K_known_x=value_or_func(K_known_x, self._compute_K_known_x, x),
        )

    def _mean_default(
        self, x: numpy.ndarray, K_x_known: numpy.ndarray
    ) -> numpy.ndarray:
        if self.x_known is None:  # GP is Prior
            return self.prior_mean(x)

        mean = (
            self.prior_mean(x)
            + K_x_known @ self.K_known_known_inv @ self.condition_values
        )

        return mean

    def _mean__covariance_kernel(
        self, x: numpy.ndarray, K_x_known: numpy.ndarray
    ) -> numpy.ndarray:
        if self.x_known is None:  # GP is Prior
            return self.prior_mean(x)

        return self.prior_mean(x) + K_x_known @ self.RW06_2_1_alpha

    def mean(self, x: numpy.ndarray, K_x_known: numpy.ndarray = None) -> numpy.ndarray:
        K_x_known = value_or_func(K_x_known, self._compute_K_x_known, x)
        if self.kernel.is_covariance_function:
            return self._mean__covariance_kernel(x, K_x_known=K_x_known)
        return self._mean_default(x, K_x_known=K_x_known)

    def _covariance_default(
        self,
        K_x_x: numpy.ndarray,
        K_x_known: numpy.ndarray,
        K_known_x: numpy.ndarray,
    ) -> numpy.ndarray:
        if self.x_known is None:  # GP is Prior
            return K_x_x

        return K_x_x - K_x_known @ self.K_known_known_inv @ K_known_x

    def _covariance_symmetric_kernel(
        self, K_x_x: numpy.ndarray, K_x_known: numpy.ndarray
    ) -> numpy.ndarray:
        if self.x_known is None:  # GP is Prior
            return K_x_x

        return K_x_x - K_x_known @ self.K_known_known_inv @ K_x_known.T

    def _covariance__covariance_kernel(
        self, K_x_x: numpy.ndarray, K_x_known: numpy.ndarray
    ) -> numpy.ndarray:
        if self.x_known is None:  # GP is Prior
            return K_x_x

        v = scipy.linalg.solve_triangular(
            self.K_known_known_cholesky, K_x_known.T, lower=True
        )

        return K_x_x - v.T @ v

    def covariance(
        self,
        x: numpy.ndarray,
        K_x_x: numpy.ndarray = None,
        K_x_known: numpy.ndarray = None,
        K_known_x: numpy.ndarray = None,
    ) -> numpy.ndarray:
        K_x_x = value_or_func(K_x_x, self._compute_K_x_x, x)
        K_x_known = value_or_func(K_x_known, self._compute_K_x_known, x)
        if self.kernel.is_covariance_function:
            return self._covariance__covariance_kernel(K_x_x=K_x_x, K_x_known=K_x_known)
        if self.kernel.is_symmetric:
            return self._covariance_symmetric_kernel(K_x_x=K_x_x, K_x_known=K_x_known)
        return self._covariance_default(
            K_x_x=K_x_x,
            K_x_known=K_x_known,
            K_known_x=value_or_func(K_known_x, self._compute_K_known_x, x),
        )

    def variance(
        self,
        x: numpy.ndarray,
        covariance: numpy.ndarray = None,
    ) -> numpy.ndarray:
        return self.np.diag(value_or_func(covariance, self.covariance, x))

    def std_deviation(
        self, x: numpy.ndarray, variance: numpy.ndarray = None
    ) -> numpy.ndarray:
        return self.np.sqrt(
            self.np.abs(value_or_func(variance, self.variance, x))
        )  # abs in case of small error or kernel not pd

    def __call__(self, x: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        """See :func:`<GaussianProcess.evaluate>`"""
        return self.evaluate(x)

    def _derivative_default(
        self,
        x: numpy.ndarray,
        K_x_x_derivative_lhs: numpy.ndarray,
        K_x_known: numpy.ndarray,
        K_known_x: numpy.ndarray,
        K_x_known_derivative_lhs: numpy.ndarray,
        K_known_x_derivative_rhs: numpy.ndarray,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Evaluate the mean and covariance matrix of the derivative of the Gaussian Process

        Uses the standard evaluation formula for Gaussian Process posteriors

        Args:
            x (ndarray): The point(s) at which to evaluate the mean and covariance matrix

        Returns:
            (ndarray, ndarray): mean and covariance matrix at x
        """
        return self._derivative_mean_default(
            x, K_x_known_derivative_lhs=K_x_known_derivative_lhs
        ), self.np.diag(
            self._derivative_covariance_default(
                K_x_x_derivative_lhs=K_x_x_derivative_lhs,
                K_x_known=K_x_known,
                K_known_x=K_known_x,
                K_x_known_derivative_lhs=K_x_known_derivative_lhs,
                K_known_x_derivative_rhs=K_known_x_derivative_rhs,
            )
        )

    def _derivative_symmetric_kernel(
        self,
        x: numpy.ndarray,
        K_x_x_derivative_lhs: numpy.ndarray,
        K_x_known: numpy.ndarray,
        K_x_known_derivative_lhs: numpy.ndarray,
    ):
        return self._derivative_mean_default(
            x, K_x_known_derivative_lhs=K_x_known_derivative_lhs
        ), self.np.diag(
            self._derivative_covariance_symmetric_kernel(
                K_x_x_derivative_lhs=K_x_x_derivative_lhs,
                K_x_known=K_x_known,
                K_x_known_derivative_lhs=K_x_known_derivative_lhs,
            )
        )

    def _derivative__covariance_kernel(
        self,
        x: numpy.ndarray,
        K_x_x_derivative_lhs: numpy.ndarray,
        K_x_known: numpy.ndarray,
        K_x_known_derivative_lhs: numpy.ndarray,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Evaluate the mean and covariance matrix of the derivative of the Gaussian Process

        Uses cholesky decomposition to avoid calculating the inverse of K_known_known offering a more stable evaluation

        Args:
            x (ndarray): The point(s) at which to evaluate the mean and covariance matrix

        Returns:
            (ndarray, ndarray): mean and covariance matrix at x
        """
        return self._derivative_mean__covariance_kernel(
            x, K_x_known_derivative_lhs=K_x_known_derivative_lhs
        ), self.np.diag(
            self._derivative_covariance__covariance_kernel(
                K_x_x_derivative_lhs=K_x_x_derivative_lhs,
                K_x_known=K_x_known,
                K_x_known_derivative_lhs=K_x_known_derivative_lhs,
            )
        )

    def derivative(
        self,
        x: numpy.ndarray,
        K_x_x_derivative_lhs: numpy.ndarray = None,
        K_x_known: numpy.ndarray = None,
        K_known_x: numpy.ndarray = None,
        K_x_known_derivative_lhs: numpy.ndarray = None,
        K_known_x_derivative_rhs: numpy.ndarray = None,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Evaluate the mean and variance of the derivative of the Gaussian Process

        Uses the default evaluation, if self is not a valid Gaussian Process and more stable evaluation via cholesky decomposition otherwise.

        Args:
            x (ndarray): The point(s) at which to evaluate the mean and covariance matrix

        Returns:
            (ndarray, ndarray): derivative of mean and its variance at x
        """

        K_x_x_derivative_lhs = value_or_func(
            K_x_x_derivative_lhs, self._compute_K_x_x_derivative_lhs, x
        )
        K_x_known = value_or_func(K_x_known, self._compute_K_x_known, x)
        K_x_known_derivative_lhs = value_or_func(
            K_x_known_derivative_lhs, self._compute_K_x_known_derivative_lhs, x
        )

        if self.kernel.is_positive_definite:
            return self._derivative__covariance_kernel(
                x,
                K_x_x_derivative_lhs=K_x_x_derivative_lhs,
                K_x_known=K_x_known,
                K_x_known_derivative_lhs=K_x_known_derivative_lhs,
            )
        if self.kernel.is_symmetric:
            return self._derivative_symmetric_kernel(
                x,
                K_x_x_derivative_lhs=K_x_x_derivative_lhs,
                K_x_known=K_x_known,
                K_x_known_derivative_lhs=K_x_known_derivative_lhs,
            )
        return self._derivative_default(
            x,
            K_x_x_derivative_lhs=K_x_x_derivative_lhs,
            K_x_known=K_x_known,
            K_known_x=value_or_func(K_known_x, self._compute_K_known_x, x),
            K_x_known_derivative_lhs=K_x_known_derivative_lhs,
            K_known_x_derivative_rhs=value_or_func(
                K_known_x_derivative_rhs, self._compute_K_known_x_derivative_rhs, x
            ),
        )

    def _derivative_mean_default(
        self, x: numpy.ndarray, K_x_known_derivative_lhs: numpy.ndarray
    ) -> numpy.ndarray:
        if self.x_known is None:  # GP is prior
            return self.prior_mean.derivative(x)

        derivative_mean = (
            self.prior_mean.derivative(x)
            + K_x_known_derivative_lhs @ self.K_known_known_inv @ self.condition_values
        )

        return derivative_mean

    def _derivative_mean__covariance_kernel(
        self, x: numpy.ndarray, K_x_known_derivative_lhs: numpy.ndarray
    ) -> numpy.ndarray:
        if self.x_known is None:  # GP is prior
            return self.prior_mean.derivative(x)

        return (
            self.prior_mean.derivative(x)
            + K_x_known_derivative_lhs @ self.RW06_2_1_alpha
        )

    def derivative_mean(
        self, x: numpy.ndarray, K_x_known_derivative_lhs: numpy.ndarray = None
    ) -> numpy.ndarray:
        K_x_known_derivative_lhs = value_or_func(
            K_x_known_derivative_lhs, self._compute_K_x_known_derivative_lhs, x
        )

        return (
            self._derivative_mean_default(x, K_x_known_derivative_lhs)
            if not self.kernel.is_covariance_function
            else self._derivative_mean__covariance_kernel(x, K_x_known_derivative_lhs)
        )

    def _derivative_covariance_default(
        self,
        K_x_x_derivative_lhs: numpy.ndarray,
        K_x_known: numpy.ndarray,
        K_known_x: numpy.ndarray,
        K_x_known_derivative_lhs: numpy.ndarray,
        K_known_x_derivative_rhs: numpy.ndarray,
    ) -> numpy.ndarray:
        if self.x_known is None:  # GP is prior
            return K_x_x_derivative_lhs

        return (
            K_x_x_derivative_lhs
            - K_x_known_derivative_lhs @ self.K_known_known_inv @ K_known_x
            - K_x_known @ self.K_known_known_inv @ K_known_x_derivative_rhs
        )

    def _derivative_covariance_symmetric_kernel(
        self,
        K_x_x_derivative_lhs: numpy.ndarray,
        K_x_known: numpy.ndarray,
        K_x_known_derivative_lhs: numpy.ndarray,
    ):
        if self.x_known is None:  # GP is prior
            return K_x_x_derivative_lhs

        return (
            K_x_x_derivative_lhs
            - 2.0 * K_x_known_derivative_lhs @ self.K_known_known_inv @ K_x_known.T
        )

    def _derivative_covariance__covariance_kernel(
        self,
        K_x_x_derivative_lhs: numpy.ndarray,
        K_x_known: numpy.ndarray,
        K_x_known_derivative_lhs: numpy.ndarray,
    ) -> numpy.ndarray:
        if self.x_known is None:  # GP is prior
            return K_x_x_derivative_lhs

        v_l = scipy.linalg.solve_triangular(
            self.K_known_known_cholesky, K_x_known_derivative_lhs.T, lower=True
        )
        v_r = scipy.linalg.solve_triangular(
            self.K_known_known_cholesky, K_x_known.T, lower=True
        )

        variance = K_x_x_derivative_lhs - 2.0 * v_l.T @ v_r

        return variance

    def derivative_covariance(
        self,
        x: numpy.ndarray,
        K_x_x_derivative_lhs: numpy.ndarray = None,
        K_x_known: numpy.ndarray = None,
        K_known_x: numpy.ndarray = None,
        K_x_known_derivative_lhs: numpy.ndarray = None,
        K_known_x_derivative_rhs: numpy.ndarray = None,
    ) -> numpy.ndarray:
        K_x_x_derivative_lhs = value_or_func(
            K_x_x_derivative_lhs, self._compute_K_x_x_derivative_lhs, x
        )
        K_x_known = value_or_func(K_x_known, self._compute_K_x_known, x)
        K_x_known_derivative_lhs = value_or_func(
            K_x_known_derivative_lhs, self._compute_K_x_known_derivative_lhs, x
        )

        if self.kernel.is_positive_definite:
            return self._derivative_covariance__covariance_kernel(
                K_x_x_derivative_lhs=K_x_x_derivative_lhs,
                K_x_known=K_x_known,
                K_x_known_derivative_lhs=K_x_known_derivative_lhs,
            )
        if self.kernel.is_symmetric:
            return self._derivative_covariance_symmetric_kernel(
                K_x_x_derivative_lhs=K_x_x_derivative_lhs,
                K_x_known=K_x_known,
                K_x_known_derivative_lhs=K_x_known_derivative_lhs,
            )
        return self._derivative_covariance_default(
            K_x_x_derivative_lhs=K_x_x_derivative_lhs,
            K_x_known=K_x_known,
            K_known_x=value_or_func(K_known_x, self._compute_K_known_x, x),
            K_x_known_derivative_lhs=K_x_known_derivative_lhs,
            K_known_x_derivative_rhs=value_or_func(
                K_known_x_derivative_rhs, self._compute_K_known_x_derivative_rhs, x
            ),
        )

    def derivative_variance(
        self, x: numpy.ndarray, derivative_covariance: numpy.ndarray = None
    ) -> numpy.ndarray:
        return self.np.diag(
            value_or_func(derivative_covariance, self.derivative_covariance, x)
        )

    def derivative_std_deviation(
        self, x: numpy.ndarray, derivative_variance: numpy.ndarray = None
    ) -> numpy.ndarray:
        return self.np.sqrt(
            self.np.abs(value_or_func(derivative_variance, self.derivative_variance, x))
        )  # abs in case of small error or kernel not pd

    def std_deviation_derivative(
        self,
        x: numpy.ndarray,
        derivative_variance: numpy.ndarray = None,
        std_deviation: numpy.ndarray = None,
    ) -> numpy.ndarray:
        """The derivative of the std deviation

        Args:
            x (numpy.ndarray): The point(s) at which to evaluate the derivative of the std deviation

        Returns:
            numpy.ndarray: The std deviation at x
        """
        derivative_variance = value_or_func(
            derivative_variance, self.derivative_variance, x
        )
        std_deviation = value_or_func(std_deviation, self.std_deviation, x)

        return self.np.divide(
            derivative_variance, 2.0 * std_deviation, where=(std_deviation != 0.0)
        )

    def sample(
        self,
        rng,
        x: numpy.ndarray,
        mean: numpy.ndarray = None,
        covariance: numpy.ndarray = None,
    ) -> numpy.ndarray:
        """Sample the Gaussian Process at X

        Args:
            rng (Generator): The random number generator to use
            X (ndarray): The point(s) at which to sample

        Returns:
            ndarray: The function values of the sampled function at X
        """
        mean = value_or_func(mean, self.mean, x)
        covariance = value_or_func(covariance, self.covariance, x)

        return rng.multivariate_normal(mean, covariance).T

    def log_marginal_likelihood_default(self) -> float:
        """Computes the log marginal likelihood of this Gaussian Process

        Uses the standard evaluation formula that calculates the determinant

        Returns:
            The log marginal likelihood
        """

        t_0 = (
            0.5
            * self.condition_values.T
            @ self.K_known_known_inv
            @ self.condition_values
        )
        t_1 = 0.5 * self.np.log(self.np.linalg.det(self.K_known_known))
        t_2 = len(self.condition_values) * 0.5 * self.np.log(2 * self.np.pi)
        return -(t_0 + t_1 + t_2)

    def log_marginal_likelihood_stable(self) -> float:
        """Computes the log marginal likelihood of this Gaussian Process

        Uses cholesky decomposition to avoid calculating the determinant of K_known_known offering a more stable evaluation

        Returns:
            The log marginal likelihood
        """

        t_0 = 0.5 * self.condition_values.T @ self.RW06_2_1_alpha
        t_1 = self.np.sum(self.np.log(self.np.diag(self.K_known_known_cholesky)))
        t_2 = len(self.condition_values) * 0.5 * self.np.log(2 * self.np.pi)
        return -(t_0 + t_1 + t_2)

    def log_marginal_likelihood(self) -> float:
        """Computes the log marginal likelihood of this Gaussian Process

        Uses the default evaluation, if self is not a valid Gaussian Process and more stable evaluation via cholesky decomposition otherwise.

        Returns:
            The log marginal likelihood
        """
        return (
            self.log_marginal_likelihood_default()
            if not self.kernel.is_covariance_function
            else self.log_marginal_likelihood_stable()
        )

    def f_error(self) -> float:
        """Computes the absolute error of the function values at the known positions."""
        gp_f_values, _ = self.evaluate(self.x_known)
        return self.np.abs(self.f_known - gp_f_values).sum()

    def g_error(self) -> float:
        """Computes the absolute error of the gradient values at the known positions"""
        gp_g_values, _ = self.derivative(self.x_known)
        return self.np.abs(self.g_known - gp_g_values).sum()
