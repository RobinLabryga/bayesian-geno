from gaussian_process import GaussianProcess, GPPrediction
from gaussian_process.kernels import Matern2_5Kernel
import gaussian_process.GPfunctions as gp
import numpy
import types
from scipy import stats
from acquisition import AcquisitionFunction, LowerConfidenceBound
from acquisition.optimization import DIRECT_LBFGSB_AcquisitionOptimizer
from gaussian_process.prior_mean import ConstantMean
from dataclasses import dataclass
import warnings

from util import value_or_value, value_or_func


class LineSearchDebugOptions:
    def __init__(
        self,
        report_termination_reason: bool = False,
        report_wolfe_termination: bool = False,
        report_return_value: bool = False,
        report_insufficient_acquisition: bool = False,
        report_invalid_f: bool = False,
        report_acquisition_max: bool = False,
        report_area_reduction: bool = False,
        report_kernel_hyperparameter: bool = False,
        gp_verbose: bool = False,
        plot_gp: bool = False,
        plot_threshold: int = numpy.inf,
        report_clipping: bool = False,
    ) -> None:
        """A structure encapsulating options about line search debug

        Args:
            report_termination_reason (bool, optional): False if no info about the reason for termination should be reported. Defaults to False.
            report_termination_reason (bool, optional): True to report if a point satisfying the wolfe condition is returned. Defaults to False
            report_return_value (bool, optional): True if the step and function value that are being returned should be reported. Defaults to False.
            report_insufficient_acquisition (bool, optional): False if no info about failure to find unique new step should be reported. Defaults to False.
            report_invalid_f (bool, optional): False if no info about invalid function values (inf, nan) should be reported. Defaults to False.
            report_acquisition_max (bool, optional): True if the found maximum of the acquisition function should be reported. Defaults to False
            gp_verbose (bool, optional): True if Gaussian Process creation should happen verbosely. Defaults to False.
            plot_gp (bool, optional): True if the Gaussian Process and acquisition function and objective should be plotted in each iteration. Defaults to False.
            plot_threshold (int, optional). The iteration after which every iteration should be plotted. Ignored if plot_gp is False. The last iteration is plotted as long as plot_pg is True. Defaults to 1000.
        """
        self.report_termination_reason = report_termination_reason
        self.report_wolfe_termination = report_wolfe_termination
        self.report_return_value = report_return_value
        self.report_insufficient_acquisition = report_insufficient_acquisition
        self.report_invalid_f = report_invalid_f
        self.report_acquisition_max = report_acquisition_max
        self.report_area_reduction = report_area_reduction
        self.report_kernel_hyperparameter = report_kernel_hyperparameter
        self.gp_verbose = gp_verbose
        self.plot_gp = plot_gp
        self.plot_threshold = plot_threshold
        self.report_clipping = report_clipping
        # TODO: Options to disable acquisition, objective, gp, derivatives


def init_debug(search_interval, fg, np):
    S_debug = np.linspace(start=search_interval[0], stop=search_interval[1], num=100)
    f_debug = np.array([fg(s, debug=True)[0] for s in S_debug])
    return S_debug, f_debug


def print_debug_info(
    GP_posterior: GaussianProcess,
    S_debug: numpy.ndarray,
    f_debug: numpy.ndarray,
    step_known: numpy.ndarray,
    f_known: numpy.ndarray,
    acquisitionFunction: AcquisitionFunction,
) -> None:
    import matplotlib.pyplot as plt

    if GP_posterior is None:
        fig, ax1 = plt.subplots(1, 1, sharex=True)
        gp.plot_objective(ax1, S_debug, f_debug, step_known, f_known)

        gp.plot_label(ax1, f"{len(step_known)} data points")
        plt.show()
        return

    assert acquisitionFunction is not None

    pred = GPPrediction(S_debug, GP_posterior)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    gp.plot_objective(ax1, S_debug, f_debug, step_known, f_known)
    gp.plot_gp(ax1, S_debug, pred.mean, pred.std_deviation, drawStd=False)
    gp.plot_observations(ax2, step_known, f_known)
    gp.plot_gp(ax2, S_debug, pred.mean, pred.std_deviation)

    ax3.plot(S_debug, [acquisitionFunction(s) for s in S_debug], label="Acquisition")
    # ax1.plot(
    #     S_debug,
    #     [acquisitionFunction.derivative(s) for s in S_debug],
    #     label="Acquisition Gradient",
    # )

    gp.plot_label(ax1, f"{len(step_known)} data points")
    gp.plot_label(ax2, None)
    gp.plot_label(ax3, None)
    plt.show()


@dataclass
class DataPoint:
    step: float
    x: numpy.ndarray
    f: float
    g: numpy.ndarray

    def __str__(self) -> str:
        return f"s={self.step}, f={self.f}"

    def __repr__(self) -> str:
        return self.__str__()


class LineSearchFunctionWrapper:
    def __init__(
        self,
        fg,
        x0: numpy.ndarray,
        f0: float,
        g0: numpy.ndarray,
        d: numpy.ndarray,
        step_min: float,
        step_max: float,
        np: types.ModuleType = None,
        wolfe_c1: float = 1.0e-4,
        wolfe_c2: float = 0.9,
    ) -> None:
        """A class wrapping function access for line search

        Args:
            fg (_type_): The objective function
            x0 (numpy.ndarray): The search start point
            f0 (float): The function value at x0
            g0 (numpy.ndarray): The gradient at x0
            d (numpy.ndarray): The search direction
            step_min (float): The left bound of the interval
            step_max (float): The right bound of the interval
            np (types.ModuleType, optional): The numpy module to use. numpy on None Defaults to None.
            wolfe_c1 (float, optional): The parameter for the sufficient decrease condition. Defaults to 1.0e-4.
            wolfe_c2 (float, optional): The parameter for the curvature condition. Defaults to 0.9.
        """
        assert step_min <= 0.0 <= step_max

        self.np = value_or_value(np, numpy)
        self.__fg = fg
        self.x0 = x0
        self.f0 = f0
        self.dg0 = d.T @ g0
        self.d = d
        self.step_min = step_min
        self.step_max = step_max
        self.__wolfe_c1 = wolfe_c1
        self.__wolfe_c2 = wolfe_c2
        self.__data_points = {0.0: DataPoint(0.0, x0, f0, g0)}
        self.fun_eval = 0

        self.step_best = 0.0
        self.x_best = x0
        self.f_best = f0
        self.g_best = g0
        self.dg_best = self.dg0

    def data_point(self, step: float) -> DataPoint:
        """
        Args:
            step (float): The step

        Returns:
            DataPoint: The data point at step
        """
        assert self.step_min <= step <= self.step_max
        assert step in self.__data_points
        return self.__data_points[step]

    def update_step_bounds(self, step_min: float, step_max: float):
        assert step_min <= step_max, f"{step_min} > {step_max}"

        if not (step_min <= self.step_best <= step_max) and self.fg(step_min)[0] > self.f_best and self.fg(step_max)[0] > self.f_best:
            if not ((self.x_best == self.x(step_min)).all() or (self.x_best == self.x(step_max)).all()):
                warnings.warn(f"Best step {self.step_best} (f={self.f_best}) outside interval {step_min} (f={self.fg(step_min)[0]}) to {step_max} (f={self.fg(step_max)[0]})")

        self.step_min = step_min
        self.step_max = step_max

        self.__data_points = {step: data_point for step, data_point in self.__data_points.items() if step_min <= step <= step_max}

    def x(self, step: float) -> numpy.ndarray:
        """x0 + step * d

        Args:
            step (float): The step

        Returns:
            numpy.ndarray: The x value at step
        """
        return self.x0 + step * self.d

    def fg(self, step: float, debug: bool = False) -> tuple[float, numpy.ndarray]:
        """
        Args:
            step (float): The step

        Returns:
            tuple[float, numpy.ndarray]: function value and gradient at step
        """
        if debug:
            return self.__fg(self.x(step))

        assert self.step_min <= step <= self.step_max, f"{self.step_min}, {step}, {self.step_max}"

        if step not in self.__data_points:
            # TODO: Check if x already exists to avoid duplicate evaluation for case where step too small to change x numerically
            x = self.x(step)
            f, g = self.__fg(x)
            self.fun_eval += 1
            self.__data_points[step] = DataPoint(step, x, f, g)
            if (
                f < self.f_best
                or (
                    f == self.f_best
                    and step > self.step_best
                    and (
                        self.x_best != x
                    ).any()  # Step too small to change x numerically
                )
            ) and self.np.isfinite(g).all():
                self.f_best = f
                self.g_best = g
                self.x_best = x
                self.dg_best = self.d.T @ g
                self.step_best = step
        data_point = self.__data_points[step]
        return data_point.f, data_point.g

    # TODO: Store phi and psi with datapoints to avoid duplicate evaluations and enable iteration over all datapoints
    def phi(self, step: float, debug: bool = False) -> tuple[float, float]:
        """
        Args:
            step (float): The step

        Returns:
            tuple[float, float]: The function value and direction gradient at step
        """
        f, g = self.fg(step, debug)
        return f, self.d.T @ g

    def psi(self, step: float, debug: bool = False) -> tuple[float, float]:
        """
        Args:
            step (float): The step

        Returns:
            tuple[float, float]: The difference between the sufficient decrease condition ray and the function value and the gradient of the difference at step
        """
        phi_0_f, phi_0_g = self.f0, self.dg0
        phi_f, phi_g = self.phi(step, debug)
        return (
            phi_f - (phi_0_f + self.__wolfe_c1 * self.dg0 * step),
            phi_g - self.__wolfe_c1 * phi_0_g,
        )

    def known_steps(self):
        return self.__data_points.keys()

    def sufficient_decrease_met(self, step):
        f, g = self.fg(step)
        return f <= self.f0 + self.__wolfe_c1 * step * self.dg0

    def curvature_met(self, step):
        f, g = self.fg(step)
        return -self.d.T @ g <= -self.__wolfe_c2 * self.dg0

    def modified_curvature_met(self, step):
        f, g = self.fg(step)
        return self.np.abs(self.d.T @ g) <= self.__wolfe_c2 * self.np.abs(self.dg0)

    def wolfe_condition_met(self, step):
        f, g = self.fg(step)
        return (f <= self.f0 + self.__wolfe_c1 * step * self.dg0) and (
            -self.d.T @ g <= -self.__wolfe_c2 * self.dg0
        )
        # return self.sufficient_decrease_met(step) and self.curvature_met(step)

    def strong_wolfe_condition_met(self, step):
        f, g = self.fg(step)
        return (f <= self.f0 + self.__wolfe_c1 * step * self.dg0) and (
            self.np.abs(self.d.T @ g) <= self.__wolfe_c2 * self.np.abs(self.dg0)
        )
        # return self.sufficient_decrease_met(step) and self.modified_curvature_met(step)


def find_best_step(step_known, f_known, np):
    # TODO: Maybe return best of posterior mean instead (literature has some info)

    smallest_value_indices = np.nonzero(f_known == np.nanmin(f_known))[0]
    largest_step_with_smallest_value_index = smallest_value_indices[
        np.argmax(step_known[smallest_value_indices])
    ]
    return step_known[largest_step_with_smallest_value_index]


def return_best_step(step_known, f_known, np):
    return find_best_step(step_known, f_known, np), False

def clip_step(step, x0, x1, debug_options, np):
    # Clip step within 10% if cubic edges
    if not (x0 + (x1 - x0) * .1 <= step <= x1 - (x1 - x0) * .1):
        step = np.clip(step, x0 + (x1 - x0) * .1, x1 - (x1 - x0) * .1)
        if debug_options.report_clipping:
            print('step on boundary, clipped')
    return step

def gp_line_search(
    fg,
    search_interval: tuple[float, float],
    step_known,
    wolfe_condition_met,
    np: types.ModuleType,
    debug_options: LineSearchDebugOptions,
    max_sample_count: int,
) -> tuple[float, bool]:
    """_summary_

    Args:
        fg (_type_): _description_
        search_interval (tuple[float, float]): _description_
        step_known (_type_): _description_
        wolfe_condition_met (_type_): _description_
        np (types.ModuleType): _description_
        debug_options (LineSearchDebugOptions): _description_
        max_sample_count (int): _description_

    Returns:
        tuple[float, bool]: step, wolfe_met
    """

    # TODO: Maybe require step and step max present and finite

    # Containers used for debug. Initialized just in time.
    S_debug = None
    f_debug = None

    step_min, step_max = search_interval
    assert step_min < step_max, f"{step_min} >= {step_max}"
    assert step_min in step_known and step_max in step_known

    # Vectors to hold the information we have already queried previously
    step_known, f_known, g_known = zip(*[(step, *fg(step)) for step in step_known])
    step_known = np.array(step_known)
    f_known = np.array(f_known)
    g_known = np.array(g_known)

    if not np.isfinite(f_known).all():
        if debug_options.report_invalid_f:
            print(
                f"Known fs in search interval contained invalid value f_known={f_known}"
            )
        if debug_options.report_termination_reason:
            print(f"Line search terminated due to condition value not finite")

        # Let caller decide how to change search area
        return return_best_step(step_known, f_known, np)

    if not np.isfinite(g_known).all():
        if debug_options.report_invalid_f:
            print(
                f"Known gs in search interval contained invalid value g_known={g_known}"
            )
        if debug_options.report_termination_reason:
            print(f"Line search terminated due to condition value not finite")

        # Let caller decide how to change search area
        return return_best_step(step_known, f_known, np)

    step = step_max  # We start at the max step size

    f_best = min(f_known)

    k = 0  # Count how many iterations of line search were performed

    GP_posterior = None
    acquisitionFunction = None

    while True:
        # TODO: Reuse old GP if nothing but k changed

        if step in step_known:
            if k != 0:
                if debug_options.report_insufficient_acquisition:
                    print("Acquisition function found same value multiple times")
                if debug_options.report_termination_reason:
                    print(
                        "Line search terminated due to duplicate condition value found"
                    )
                break

            s_index = np.where(step_known == step)[0][0]
            f = f_known[s_index]
            step_g = g_known[s_index]
        else:  # New step
            f, step_g = fg(step)

            if not np.isfinite(f):
                if debug_options.report_invalid_f:
                    print(
                        f"Encountered f={f} at step={step}, which can not be used for Gaussian Process"
                    )
                if debug_options.report_termination_reason:
                    print(f"Line search terminated due to condition value not finite")
                break  # Let caller decide how to change search area
            elif not np.isfinite(step_g):
                if debug_options.report_invalid_f:
                    print(
                        f"Encountered step_g={step_g} at step={step}, which can not be used for Gaussian Process. g={g}"
                    )
                if debug_options.report_termination_reason:
                    print(f"Line search terminated due to condition value not finite")
                break
            else:
                # Update known information
                step_known = np.append(step_known, step)
                f_known = np.append(f_known, f)
                g_known = np.append(g_known, step_g)

        # Quit if any of the quit conditions are met
        if f <= f_best and wolfe_condition_met(step):
            if debug_options.report_termination_reason:
                print(
                    f"Line search terminated due to strong Wolfe condition after {k} sample iterations with {step}"
                )
            if debug_options.plot_gp:
                if S_debug is None:
                    S_debug, f_debug = init_debug(search_interval, fg, np)
                print_debug_info(
                    GP_posterior,
                    S_debug,
                    f_debug,
                    step_known,
                    f_known,
                    acquisitionFunction,
                )
            return step, True

        if len(step_known) > max_sample_count:
            if debug_options.report_termination_reason:
                print("Line search terminated due to exceeded sample count")
            break

        if k > max_sample_count:  # TODO: Will this ever execute?
            if debug_options.report_termination_reason:
                print("Line search terminated due to exceeded iteration count")
            break

        f_best = min(f_best, f)

        # The prior mean of the Gaussian Process
        prior_mean = ConstantMean(f_best, np)

        # TODO: Consider hyperparameter optimization (log marginal likelihood)
        # Using average distance, min distance, more as length scale
        length_scales = [
            # np.min([np.abs(a - b) for a, b in itertools.pairwise(sorted(step_known))]),
            # (step_max - step_min) / (len(step_known) - 1),
            # statistics.mode([abs(a - b) for a, b in itertools.pairwise(step_known)])
            step_max
            - step_min,
        ]

        # Find length scale with best log marginal likelihood
        GP_posterior = None
        GP_posterior_lml = None
        for l in length_scales:
            # The kernel used for the GP
            kernel = Matern2_5Kernel(l=l)

            l_posterior = None
            # Compute new posterior
            noise = 1e-14 * (step_max - step_min)  # Initial noise relative to x
            while True:
                try:
                    l_posterior = GaussianProcess(
                        kernel=kernel,
                        x_known=step_known,
                        f_known=f_known,
                        g_known=g_known,
                        f_noise=noise,
                        g_noise=noise,
                        prior_mean=prior_mean,
                        np=np,
                        verbose=debug_options.gp_verbose,
                    )
                except np.linalg.LinAlgError:
                    # Numerical instability may result in covariance matrix not being positive definite. Adding more noise may fix that
                    noise *= 10
                else:
                    l_posterior_lml = l_posterior.log_marginal_likelihood()
                    GP_posterior, GP_posterior_lml = (
                        (l_posterior, l_posterior_lml)
                        if GP_posterior is None or l_posterior_lml > GP_posterior_lml
                        else (GP_posterior, GP_posterior_lml)
                    )
                    break  # Success

        if debug_options.report_kernel_hyperparameter:
            print(f"kernel with l={GP_posterior.kernel.l} with {GP_posterior_lml}")

        # Compute acquisition function
        acquisitionFunction = LowerConfidenceBound(GP_posterior, lcb_factor=2.0, np=np)

        # New step is step size with max acquisition
        acquisitionOptimizer = DIRECT_LBFGSB_AcquisitionOptimizer()
        step = acquisitionOptimizer.maximize(
            acquisitionFunction,
            step_min,
            step_max,
            step_known,
        )

        if debug_options.report_acquisition_max:
            print(f"Maximized acquisition function at {step}")
        if debug_options.plot_gp and k >= debug_options.plot_threshold:
            if S_debug is None:
                S_debug, f_debug = init_debug(search_interval, fg, np)
            print_debug_info(
                GP_posterior,
                S_debug,
                f_debug,
                step_known,
                f_known,
                acquisitionFunction,
            )

        known_step_to_left = step_min if step == step_min else max([s for s in step_known if s < step])
        known_step_to_right = step_max if step == step_max else min([s for s in step_known if s > step])
        step = clip_step(step, known_step_to_left, known_step_to_right, debug_options, np)

        k += 1

    if debug_options.plot_gp and len(step_known) > 1:
        if S_debug is None:
            S_debug, f_debug = init_debug(search_interval, fg, np)
        print_debug_info(
            GP_posterior,
            S_debug,
            f_debug,
            step_known,
            f_known,
            acquisitionFunction,
        )

    return return_best_step(step_known, f_known, np)


def get_next_interval(objective, step_l, step_u, step_t, can_guarantee_wolfe_step, np):
    f, g = objective(step_t)
    if not np.isfinite(f) or f > objective(step_l)[0]:
        return step_l, step_t, True
    else:
        assert g != 0.0

        if g * (step_l - step_t) > 0:
            return step_t, step_u, can_guarantee_wolfe_step
        else:
            return step_t, step_l, True


def update_line_search_objective(
    line_search_function: LineSearchFunctionWrapper, step, current_line_search_objective
):
    """Change objective to phi if conditions are met"""
    return (
        line_search_function.phi
        if current_line_search_objective is line_search_function.psi
        and line_search_function.psi(step)[0] <= 0.0
        and line_search_function.phi(step)[1] > 0
        else current_line_search_objective
    )


def line_search(
    x_old,
    d,
    fg,
    max_step,
    f_old,
    g_old,
    quadratic: bool = False,
    np: types.ModuleType = None,
    debug_options: LineSearchDebugOptions = None,
    max_iter: int = 1000,
    max_sample_count: int = 50,
) -> any:
    """Searches for the next step in the direction d

    Args:
        x_old (_type_): The current x value
        d (_type_): The direction to search towards
        fg (_type_): _comment_
        max_step (_type_): The maximum step size that is allowed. step will be in [0.0, min(1.0, max_step)]
        f_old (_type_): The function value at x_old
        g_old (_type_): The gradient at x_old
        quadratic (_type_, optional): True, if the function is quadratic in direction d. Defaults to False.
        np (_type_, optional): The numpy module to use. None for default numpy. Defaults to None.
        debug_options (LineSearchDebugOptions, optional): _comment_
        max_iter (int, optional): The maximum number of line search iterations that should be performed. Defaults to 1000.
        max_sample_count (int, optional): The maximum number of samples a Gaussian Process should be conditioned on. Defaults to 50.

    Returns:
        f (_type_): The function value at x
        g (_type_): The gradient at x
        x (_type_): The new x value at x_old + step * d
        step (_type_): The step length of the line search
        fun_eval (_type_): The number of function evaluations done during the line search
    """

    assert f_old is not None
    assert g_old is not None

    np = value_or_value(np, numpy)
    debug_options = value_or_func(debug_options, LineSearchDebugOptions)

    # Test to ensure that d is a descent direction
    if dg := d.T @ g_old >= 0:
        assert (
            False
        ), f"Descent direction should be descent direction. Directional gradient was {dg} in direction {d}"

    k = 0
    step = None

    step_l, step_u = 0.0, min(1.0, max_step)

    line_search_function = LineSearchFunctionWrapper(fg, x_old, f_old, g_old, d, step_l, max_step, np=np)

    if line_search_function.strong_wolfe_condition_met(step_u):
        if debug_options.report_wolfe_termination:
            print(f"Wolfe after {k} iterations")
        data_point = line_search_function.data_point(step_u)
        assert data_point.f <= line_search_function.f_best, f"Trying to return step with strong Wolfe that is not best. step={step_u}, f={data_point.f}, step_best={line_search_function.step_best} f_best={line_search_function.f_best}"
        return (
            data_point.f,
            data_point.g,
            data_point.x,
            step_u,
            line_search_function.fun_eval,
        )

    can_guarantee_wolfe_step = False

    # Phase 1: Double the right interval endpoint, until larger than step_max or strong wolfe step garanteed
    while True:
        psi_step_l_f, psi_step_l_g = line_search_function.psi(step_l)
        psi_step_u_f, psi_step_u_g = line_search_function.psi(step_u)
        if np.isnan(psi_step_u_f) or np.isnan(psi_step_u_g):
            break
        if psi_step_u_f >= psi_step_l_f or psi_step_u_g >= 0:
            can_guarantee_wolfe_step = True
            break
        if step_u >= max_step:
            break

        if line_search_function.strong_wolfe_condition_met(step_u):
            if debug_options.report_wolfe_termination:
                print(f"Wolfe after {k} iterations")
            data_point = line_search_function.data_point(step_u)
            assert data_point.f <= line_search_function.f_best, f"Trying to return step with strong Wolfe that is not best. step={step_u}, f={data_point.f}, step_best={line_search_function.step_best} f_best={line_search_function.f_best}"
            return (
                data_point.f,
                data_point.g,
                data_point.x,
                step_u,
                line_search_function.fun_eval,
            )

        if k > max_iter:
            if debug_options.report_wolfe_termination:
                print("Terminated line search due to exceeded iteration count")
            return (
                line_search_function.f_best,
                line_search_function.g_best,
                line_search_function.x_best,
                line_search_function.step_best if line_search_function.step_best != 0.0 else None,
                line_search_function.fun_eval,
            )

        k += 1

        step_l = step_u
        step_u = min(4. * step_u, max_step)

        if debug_options.report_area_reduction:
            print(f"Interval size increased to={(step_l, step_u)}")

    assert step_l <= max_step and step_u <= max_step

    line_search_function.update_step_bounds(step_l, step_u)

    # Prepopulate with steps up to sufficient decrease condition
    step = step_u
    while True:
        if line_search_function.sufficient_decrease_met(step):
            if (line_search_function.x(step) == line_search_function.x0).all():
                if debug_options.report_wolfe_termination:
                    print("Terminated line search due to step being identical to x0")
                return (
                    line_search_function.f_best,
                    line_search_function.g_best,
                    line_search_function.x_best,
                    line_search_function.step_best if line_search_function.step_best != 0.0 else None,
                    line_search_function.fun_eval,
                )
            break
        if k > max_iter:
            if debug_options.report_wolfe_termination:
                print("Terminated line search due to exceeded iteration count")
            return (
                line_search_function.f_best,
                line_search_function.g_best,
                line_search_function.x_best,
                line_search_function.step_best if line_search_function.step_best != 0.0 else None,
                line_search_function.fun_eval,
            )
        step = (9. * step_l + 1. * step) / 10.0
        k += 1

    # Phase 2: We produce sub intervals in accordance to more thuente line search to inherit convergence guarantees, while determining trial step via Bayesian optimization
    line_search_objective = update_line_search_objective(
        line_search_function, step_u, line_search_function.psi
    )

    # Only start byesian phase if step does not satisfy strong Wolfe conditions
    if (
        line_search_function.strong_wolfe_condition_met(step)
        and line_search_function.fg(step)[0] <= line_search_function.f_best # Sometimes the right hand side satisfies the strong Wolfe condition, but the left hand side does not, despite better function value.
    ):
        if debug_options.report_wolfe_termination:
            print(f"Wolfe met pre Bayesian")
        data_point = line_search_function.data_point(step)
        assert data_point.f <= line_search_function.f_best, f"Trying to return step with strong Wolfe that is not best. step={step}, f={data_point.f}, step_best={line_search_function.step_best} f_best={line_search_function.f_best}"
        return (
            data_point.f,
            data_point.g,
            data_point.x,
            step,
            line_search_function.fun_eval,
        )

    previous_step = 0.0

    interval_towards_step_max_factor = 1.1
    interval_size_decrease_factor = 2.0 / 3.0
    interval_size_prev_prev = np.inf
    interval_size_prev = np.inf

    while True:
        # Make sure interval size has decreased sufficiently in the previous iterations
        interval_size = abs(step_l - step_u)
        if interval_size >= interval_size_decrease_factor * interval_size_prev_prev:
            if debug_options.report_area_reduction:
                print(
                    f"Forced interval bisection on ({min(step_l, step_u)} ,{max(step_l, step_u)})"
                )
            step = (step_l + step_u) / 2.0
            line_search_objective = update_line_search_objective(
                line_search_function, step, line_search_objective
            )
            if debug_options.plot_gp:
                print_debug_info(
                    None,
                    *init_debug(
                        (min(step_l, step_u), max(step_l, step_u)),
                        line_search_objective,
                        np,
                    ),
                    [min(step_l, step_u), step, max(step_l, step_u)],
                    [
                        line_search_objective(min(step_l, step_u))[0],
                        line_search_objective(step)[0],
                        line_search_objective(max(step_l, step_u))[0],
                    ],
                    None,
                )
            step_l, step_u, can_guarantee_wolfe_step = get_next_interval(
                line_search_objective, step_l, step_u, step, can_guarantee_wolfe_step, np
            )
            interval_size = abs(step_l - step_u)
            k += 1
        interval_size_prev_prev, interval_size_prev = interval_size_prev, interval_size

        # Choose trial point
        step, wolfe_met = gp_line_search(
            line_search_function.phi,
            (min(step_l, step_u), max(step_l, step_u)),
            line_search_function.known_steps(),
            line_search_function.strong_wolfe_condition_met,
            np,
            debug_options,
            max_sample_count,
        )

        if debug_options.report_return_value:
            print(f"returned step={step} with f={line_search_function.fg(step)[0]}")

        # Stop if any of the termination conditions are met
        if wolfe_met:
            if debug_options.report_wolfe_termination:
                print(f"Wolfe after {k} iterations")
            data_point = line_search_function.data_point(step)
            assert data_point.f <= line_search_function.f_best, f"Trying to return step with strong Wolfe that is not best. step={step}, f={data_point.f}, step_best={line_search_function.step_best} f_best={line_search_function.f_best}"
            return (
                data_point.f,
                data_point.g,
                data_point.x,
                step,
                line_search_function.fun_eval,
            )

        if k > max_iter:
            if debug_options.report_wolfe_termination:
                print("Terminated line search due to exceeded iteration count")
            return (
                line_search_function.f_best,
                line_search_function.g_best,
                line_search_function.x_best,
                line_search_function.step_best if line_search_function.step_best != 0.0 else None,
                line_search_function.fun_eval,
            )
        k += 1

        step_data_point = line_search_function.data_point(step)

        if step_data_point.f == -np.inf:
            if debug_options.report_wolfe_termination:
                print("Terminated line search due to -inf")
            return (
                step_data_point.f,
                step_data_point.g,
                step_data_point.x,
                step_data_point.step,
                line_search_function.fun_eval,
            )

        x = step_data_point.x

        # Check if x is identical to one of the bounds despite different step values
        if step != step_l and (x == line_search_function.data_point(step_l).x).all():
            step = step_l
        elif step != step_u and (x == line_search_function.data_point(step_u).x).all():
            step = step_u

        # Ensure trial step differs from step_l and step_u
        if step in (step_l, step_u):
            steps_in_interval = sorted(
                [
                    s
                    for s in line_search_function.known_steps()
                    if min(step_l, step_u) < s
                    and s < max(step_l, step_u)
                    and (x != line_search_function.data_point(step_l).x).all()
                    and (x != line_search_function.data_point(step_u).x).all()
                ]
            )

            if len(steps_in_interval) == 0:
                step = (step_l + step_u) / 2.0
            else:
                step = steps_in_interval[
                    np.argmax(stats.gaussian_kde(steps_in_interval)(steps_in_interval))
                ]

        # Ensure sufficient movement towards step_max if wolfe step can not be guaranteed
        if not can_guarantee_wolfe_step:
            step = np.clip(step, min(interval_towards_step_max_factor * previous_step, max_step), max_step)
        previous_step = step

        line_search_objective = update_line_search_objective(
            line_search_function, step, line_search_objective
        )

        step_l, step_u, can_guarantee_wolfe_step = get_next_interval(
            line_search_objective, step_l, step_u, step, can_guarantee_wolfe_step, np
        )

        if abs(step_l - step_u) < 1e-10:
            if debug_options.report_wolfe_termination:
                print("Terminated line search due to smallest interval reached")
            return (
                line_search_function.f_best,
                line_search_function.g_best,
                line_search_function.x_best,
                line_search_function.step_best if line_search_function.step_best != 0.0 else None,
                line_search_function.fun_eval,
            )

        if debug_options.report_area_reduction:
            print(
                f"Could not find step. Restarting with search_interval={(step_l, step_u)}"
            )

        line_search_function.update_step_bounds(min(step_l, step_u), max(step_l, step_u))
