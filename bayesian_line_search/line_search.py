from gaussian_process import GaussianProcess, GPPrediction
from gaussian_process.kernels import Matern2_5Kernel
import gaussian_process.GPfunctions as gp
import numpy
import types
from scipy import stats
from acquisition import AcquisitionFunction, LowerConfidenceBound
from acquisition.optimization import DIRECT_LBFGSB_AcquisitionOptimizer
from gaussian_process.prior_mean import ZeroMean

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
        # TODO: Options to disable acquisition, objective, gp, derivatives


def normalization_parameters(f_known):
    lowest_f = min(f_known)
    largest_f = max(f_known)
    if lowest_f != largest_f:
        return -lowest_f, 1 / (largest_f - lowest_f)

    return -lowest_f, 1.0


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


class DataPoint:
    def __init__(self, step, x, f, g) -> types.NoneType:
        self.step = step
        self.x = x
        self.f = f
        self.g = g

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
            np (types.ModuleType, optional): The numpy module to use. numpy on None Defaults to None.
            wolfe_c1 (float, optional): The parameter for the sufficient decrease condition. Defaults to 1.0e-4.
            wolfe_c2 (float, optional): The parameter for the curvature condition. Defaults to 0.9.
        """
        self.np = value_or_value(np, numpy)
        self.__fg = fg
        self.x0 = x0
        self.f0 = f0
        self.dg0 = d.T @ g0
        self.d = d
        self.__wolfe_c1 = wolfe_c1
        self.__wolfe_c2 = wolfe_c2
        self.__data_points = {0.0: DataPoint(0.0, x0, f0, g0)}
        self.fun_eval = 0

    def data_point(self, step: float) -> DataPoint:
        """
        Args:
            step (float): The step

        Returns:
            DataPoint: The data point at step
        """
        assert step in self.__data_points
        return self.__data_points[step]

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

        if step not in self.__data_points:
            # TODO: Check if x already exists to avoid duplicate evaluation for case where step too small to change x numerically
            x = self.x(step)
            f, g = self.__fg(x)
            self.fun_eval += 1
            self.__data_points[step] = DataPoint(step, x, f, g)
        data_point = self.__data_points[step]
        return data_point.f, data_point.g

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

    def find_best_data_point(self):
        best = None
        for step, data_point in self.__data_points.items():
            if (
                best is None
                or data_point.f < best.f
                or (data_point.f <= best.f and data_point.step > best.step)
            ):
                best = data_point

        data_point_old = self.__data_points[0.0]

        if (data_point_old.x == best.x).all():  # step too small to change x
            return data_point_old

        return best

    def wolfe_one_met(self, step):
        f, g = self.fg(step)
        return f <= self.f0 + self.__wolfe_c1 * step * self.dg0

    def wolfe_two_met(self, step):
        f, g = self.fg(step)
        return -self.d.T @ g <= -self.__wolfe_c2 * self.dg0

    def wolfe_three_met(self, step):
        f, g = self.fg(step)
        return self.np.abs(self.d.T @ g) <= self.__wolfe_c2 * self.np.abs(self.dg0)

    def wolfe_condition_met(self, step):
        return self.wolfe_one_met(step) and self.wolfe_two_met(step)

    def strong_wolfe_condition_met(self, step):
        return self.wolfe_one_met(step) and self.wolfe_three_met(step)


def find_best_step(step_known, f_known, np):
    # TODO: Maybe return best of posterior mean instead (literature has some info)

    smallest_value_indices = np.nonzero(f_known == np.nanmin(f_known))[0]
    largest_step_with_smallest_value_index = smallest_value_indices[
        np.argmax(step_known[smallest_value_indices])
    ]
    return step_known[largest_step_with_smallest_value_index]


def return_best_step(step_known, f_known, np):
    return find_best_step(step_known, f_known, np), False


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

    step_min = search_interval[0]
    step_max = search_interval[1]
    assert step_min < step_max, f"{step_min} >= {step_max}"

    # Vectors to hold the information we have already queried previously
    step_known, f_known, g_known = zip(
        *[
            (step, *fg(step))
            for step in step_known
            if step_min <= step and step <= step_max
        ]
    )
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

    normalization_offset, normalization_scale = 0.0, 1.0

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
                    (f_debug + normalization_offset) * normalization_scale,
                    step_known,
                    (f_known + normalization_offset) * normalization_scale,
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

        # Normalize condition values to have consistent scale of std-deviation (sqrt of variance, thus scale has large effect)
        normalization_offset, normalization_scale = normalization_parameters(f_known)
        f_known_normalized = (f_known + normalization_offset) * normalization_scale
        step_g_known_normalized = g_known * normalization_scale

        # The prior mean of the Gaussian Process
        prior_mean = ZeroMean()
        # Same as ConstantMean(min(f_known), np) in un-normalized case

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
                        f_known=f_known_normalized,
                        g_known=step_g_known_normalized,
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
                (f_debug + normalization_offset) * normalization_scale,
                step_known,
                f_known_normalized,
                acquisitionFunction,
            )

        k += 1

    if debug_options.plot_gp and len(step_known) > 1:
        if S_debug is None:
            S_debug, f_debug = init_debug(search_interval, fg, np)
        print_debug_info(
            GP_posterior,
            S_debug,
            (f_debug + normalization_offset) * normalization_scale,
            step_known,
            (f_known + normalization_offset) * normalization_scale,
            acquisitionFunction,
        )

    return return_best_step(step_known, f_known, np)


def find_interval_with_wolfe(
    line_search_function, step_l, step_u, debug_options
):  # TODO: Remove
    """Moves interval to ensure point that satisfies strong Wolfe condition is inside"""
    psi_f_l, psi_g_l = line_search_function.psi(step_l)
    while True:
        psi_f_u, psi_g_u = line_search_function.psi(step_u)

        # TODO: Consider if a check for phi_g_u >= makes sense
        if psi_f_u < psi_f_l and psi_g_u < 0:
            step_u = 2 * step_u
            if debug_options.report_area_reduction:
                print(f"Moved interval to ({step_l}, {step_u})")
        else:
            return step_l, step_u


def get_next_interval(objective, step_l, step_u, step_t, np):
    f, g = objective(step_t)
    if not np.isfinite(f) or f > objective(step_l)[0]:
        return step_l, step_t
    else:
        # TODO: Make sure g is not 0.0
        if g * (step_l - step_t) > 0:
            return step_t, step_u
        else:
            return step_t, step_l


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
    step_max,
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
        step_max (_type_): The maximum step size that is allowed. step will be in [0.0, min(1.0, step_max)]
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

    # TODO: Make sure we only ever return a point that is actually the smallest we have seen, even if another one satisfies the strong Wolfe conditions.

    np = value_or_value(np, numpy)
    debug_options = value_or_func(debug_options, LineSearchDebugOptions)

    # Test to ensure that d is a descent direction
    if dg := d.T @ g_old >= 0:
        assert (
            False
        ), f"Descent direction should be descent direction. Directional gradient was {dg} in direction {d}"

    line_search_function = LineSearchFunctionWrapper(fg, x_old, f_old, g_old, d, np=np)

    k = 0
    step = None

    step_l, step_u = 0.0, 1.0

    # Phase 1: We can not ensure the presence of a strong Wolfe step and only move interval to right and increase size
    while True:
        psi_step_l_f, psi_step_l_g = line_search_function.psi(step_l)
        psi_step_u_f, psi_step_u_g = line_search_function.psi(step_u)
        if psi_step_u_f >= psi_step_l_f or psi_step_u_g >= 0:
            step_l, step_u = (
                (step_l, step_u)
                if not np.isfinite(psi_step_u_f) or psi_step_l_f <= psi_step_u_f
                else (step_u, step_l)
            )
            break

        step_t, wolfe_met = gp_line_search(
            line_search_function.psi,
            (min(step_l, step_u), max(step_l, step_u)),
            line_search_function.known_steps(),
            line_search_function.strong_wolfe_condition_met,
            np,
            debug_options,
            max_sample_count,
        )

        if debug_options.report_return_value:
            print(f"returned step={step} with f={line_search_function.fg(step)[0]}")

        if wolfe_met:
            if debug_options.report_wolfe_termination:
                print(f"Wolfe after {k} iterations")
            data_point = line_search_function.data_point(step_t)
            return (
                data_point.f,
                data_point.g,
                data_point.x,
                step_t,
                line_search_function.fun_eval,
            )

        if k > max_iter:
            if debug_options.report_wolfe_termination:
                print("Terminated line search due to exceeded iteration count")
            best_data_point = line_search_function.find_best_data_point()
            return (
                best_data_point.f,
                best_data_point.g,
                best_data_point.x,
                best_data_point.step if best_data_point.step != 0.0 else None,
                line_search_function.fun_eval,
            )

        k += 1

        step_data_point = line_search_function.data_point(step_t)

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
        if step_t != step_l and (x == line_search_function.data_point(step_l).x).all():
            assert (
                False
            ), "If the optimization returns step_t != step_u, we must have psi(step_t) <= psi(step_u) < psi(step_l)"
        elif (
            step_t != step_u and (x == line_search_function.data_point(step_u).x).all()
        ):
            step_t = step_u

        # If step_t == step_u, we move interval to right, otherwise we can guarantee strong Wolfe step in new interval
        if step_t == step_u:
            step_l, step_u = step_u, 2.0 * step_u
        else:
            psi_step_t_g = line_search_function.psi(step_t)[1]
            if psi_step_t_g > 0:
                step_l, step_u = step_t, step_l
            elif psi_step_t_g < 0:
                step_l, step_u = step_t, step_u
            else:
                assert (
                    False
                ), "psi'(step_t) can not be 0, since that implies strong Wolfe step"

            if debug_options.report_area_reduction:
                print(f"Interval size increase finished on={(step_l, step_u)}")

            break

        if debug_options.report_area_reduction:
            print(f"Interval size increased to={(step_l, step_u)}")

    # Phase 2: We can guarantee presence of strong Wolfe step and only decrease interval size
    line_search_objective = update_line_search_objective(
        line_search_function, step_u, line_search_function.psi
    )

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
            step_l, step_u = get_next_interval(
                line_search_objective, step_l, step_u, step, np
            )
            interval_size = abs(step_l - step_u)
            k += 1
        interval_size_prev_prev, interval_size_prev = interval_size_prev, interval_size

        step, wolfe_met = gp_line_search(
            line_search_objective,
            (min(step_l, step_u), max(step_l, step_u)),
            line_search_function.known_steps(),
            line_search_function.strong_wolfe_condition_met,
            np,
            debug_options,
            max_sample_count,
        )

        if debug_options.report_return_value:
            print(f"returned step={step} with f={line_search_function.fg(step)[0]}")

        if wolfe_met:
            if debug_options.report_wolfe_termination:
                print(f"Wolfe after {k} iterations")
            data_point = line_search_function.data_point(step)
            return (
                data_point.f,
                data_point.g,
                data_point.x,
                step,
                line_search_function.fun_eval,
            )

        # Stop if any of the termination conditions are met
        if k > max_iter:
            if debug_options.report_wolfe_termination:
                print("Terminated line search due to exceeded iteration count")
            best_data_point = line_search_function.find_best_data_point()
            return (
                best_data_point.f,
                best_data_point.g,
                best_data_point.x,
                best_data_point.step if best_data_point.step != 0.0 else None,
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

        line_search_objective = update_line_search_objective(
            line_search_function, step, line_search_objective
        )

        step_l, step_u = get_next_interval(
            line_search_objective, step_l, step_u, step, np
        )

        if step_l == step_u:
            if debug_options.report_wolfe_termination:
                print("Terminated line search due to smallest interval reached")
            data_point = line_search_function.data_point(step)
            return (
                data_point.f,
                data_point.g,
                data_point.x,
                step,
                line_search_function.fun_eval,
            )

        if debug_options.report_area_reduction:
            print(
                f"Could not find step. Restarting with search_interval={(step_l, step_u)}"
            )
