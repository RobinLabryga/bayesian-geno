from gaussian_process import GaussianProcess, GPPrediction
from gaussian_process.kernels import (
    Matern2_5Kernel,
)
import gaussian_process.GPfunctions as gp
import numpy
import types
from acquisition import (
    AcquisitionFunction,
    LowerConfidenceBound,
)
from acquisition.optimization import (
    DIRECT_LBFGSB_AcquisitionOptimizer,
)
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


def init_S_debug(search_interval, np):
    return np.linspace(start=search_interval[0], stop=search_interval[1], num=100)


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
    gp.plot_objective(ax2, S_debug, f_debug, step_known, f_known)
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
    def __init__(self, step, x, f, g, step_g) -> types.NoneType:
        self.step = step
        self.x = x
        self.f = f
        self.g = g
        self.step_g = step_g

    def __str__(self) -> str:
        return f"s={self.step}, f={self.f}"

    def __repr__(self) -> str:
        return self.__str__()


def find_best_data_point(data_points: list[DataPoint]):
    best = None
    for data_point in data_points:
        if best is None or data_point.f < best.f:
            best = data_point
    return data_point


def wolfe_one_met(f, step, d, f_old, g_old, c):
    return f <= f_old + c * step * d.T @ g_old


def wolfe_two_met(g, d, g_old, c):
    return -d.T @ g <= -c * d.T @ g_old


def wolfe_three_met(g, d, g_old, c, np):
    return np.abs(d.T @ g) <= c * np.abs(d.T @ g_old)


def wolfe_condition_met(f, g, step, d, f_old, g_old, np, c1=1.0e-4, c2=0.9):
    return wolfe_one_met(f, step, d, f_old, g_old, c1) and wolfe_two_met(
        g, d, g_old, c2
    )


def strong_wolfe_condition_met(f, g, step, d, f_old, g_old, np, c1=1.0e-4, c2=0.9):
    return wolfe_one_met(f, step, d, f_old, g_old, c1) and wolfe_three_met(
        g, d, g_old, c2, np
    )


def line_search_condition_met(f, f_old, step, dg):
    alpha = 0.1
    return f <= f_old + alpha * step * dg


def find_best_step(x_old, f_old, g_old, d, step_known, f_known, g_known, np):
    # TODO: Maybe return best of posterior mean instead (literature has some info)

    smallest_value_indices = np.nonzero(f_known == np.min(f_known))[0]
    largest_step_with_smallest_value_index = smallest_value_indices[
        np.argmax(step_known[smallest_value_indices])
    ]
    step = step_known[largest_step_with_smallest_value_index]

    if step == 0.0:
        return f_old, g_old, x_old, None

    x = x_old + step * d

    if (x == x_old).all():  # step too small to change x
        return f_old, g_old, x_old, None

    f = f_known[largest_step_with_smallest_value_index]
    g = g_known[largest_step_with_smallest_value_index]

    return f, g, x, step


def return_best_step(
    x_old, f_old, g_old, d, step_known, f_known, g_known, fun_eval, np, debug_options
):
    f, g, x, step = find_best_step(
        x_old, f_old, g_old, d, step_known, f_known, g_known, np
    )

    if debug_options.report_return_value:
        print(f"Best step at {step} with f={f}")

    return f, g, x, step, fun_eval, False


def gp_line_search(
    x_old,
    d,
    d_norm,
    fg,
    search_interval: tuple[float, float],
    f_old,
    g_old,
    data_points: list[DataPoint],
    np: types.ModuleType,
    debug_options: LineSearchDebugOptions,
    max_sample_count: int,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, int, bool]:
    """_summary_

    Args:
        x_old (_type_): _description_
        d (_type_): _description_
        d_norm (_type_): _description_
        fg (_type_): _description_
        search_interval (tuple[float, float]): _description_
        f_old (_type_): _description_
        g_old (_type_): _description_
        data_points (list[DataPoint]): _description_
        np (types.ModuleType): _description_
        debug_options (LineSearchDebugOptions): _description_
        max_sample_count (int): _description_

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, int, bool]: f, g, x, step, fun_eval, wolfe_met
    """

    # Containers used for debug. Initialized just in time.
    S_debug = None
    f_debug = None

    step_min = search_interval[0]
    step_max = search_interval[1]
    assert step_min < step_max  # TODO: Remove

    # Vectors to hold the information we have already queried previously
    step_known, f_known, g_known, step_g_known = zip(
        *[
            (p.step, p.f, p.g, p.step_g)
            for p in data_points
            if step_min <= p.step and p.step <= step_max
        ]
    )
    step_known = np.array(step_known)
    f_known = np.array(f_known)
    g_known = list(g_known)
    step_g_known = np.array(step_g_known)

    if not np.isfinite(f_known).all():
        if debug_options.report_invalid_f:
            print(
                f"Known fs in search interval contained invalid value f_known={f_known}"
            )
        if debug_options.report_termination_reason:
            print(f"Line search terminated due to condition value not finite")

        # Let caller decide how to change search area
        return return_best_step(
            x_old, f_old, g_old, d, step_known, f_known, g_known, 0, np, debug_options
        )

    step = step_max  # We start at the max step size

    k = 0  # Count how many iterations of line search were performed
    fun_eval = 0  # Count how many times the function was evaluated

    normalization_offset, normalization_scale = 0.0, 1.0

    GP_posterior = None
    acquisitionFunction = None

    while True:
        # TODO: Reuse old GP if nothing but k changed
        x = x_old + step * d

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
            g = g_known[s_index]
            step_g = step_g_known[s_index]
        else:  # New step
            f, g = fg(x)
            step_g = np.dot(g, d_norm)
            data_points.append(DataPoint(step, x, f, g, step_g))
            fun_eval += 1

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
                g_known.append(g)
                step_g_known = np.append(step_g_known, step_g)

        # Quit if any of the quit conditions are met
        if strong_wolfe_condition_met(f, g, step, d, f_old, g_old, np):
            # TODO: Make parameter if strong or normal wolfe condition
            if debug_options.report_termination_reason:
                print(
                    f"Line search terminated due to strong Wolfe condition after {k} sample iterations with {step}"
                )
            if debug_options.plot_gp:
                if S_debug is None:
                    S_debug = init_S_debug(search_interval, np)
                    f_debug = np.array([fg(x_old + s * d)[0] for s in S_debug])
                print_debug_info(
                    GP_posterior,
                    S_debug,
                    (f_debug + normalization_offset) * normalization_scale,
                    step_known,
                    (f_known + normalization_offset) * normalization_scale,
                    acquisitionFunction,
                )
            return f, g, x, step, fun_eval, True
        if len(step_known) > max_sample_count:
            if debug_options.report_termination_reason:
                print("Line search terminated due to exceeded sample count")
            break
        if k > max_sample_count:  # TODO: Will this ever execute?
            if debug_options.report_termination_reason:
                print("Line search terminated due to exceeded iteration count")
            break

        # Normalize condition values to have consistent scale of std-deviation (sqrt of variance, thus scale has large effect)
        normalization_offset, normalization_scale = normalization_parameters(f_known)
        f_known_normalized = (f_known + normalization_offset) * normalization_scale
        step_g_known_normalized = step_g_known * normalization_scale

        # The prior mean of the Gaussian Process
        prior_mean = ZeroMean()
        # Same as ConstantMean(min(f_known), np) in un-normalized case

        # TODO: Consider hyperparameter optimization (log marginal likelihood)
        # Using average distance, min distance, more as length scale
        length_scales = [
            # np.min([np.abs(a - b) for a, b in itertools.pairwise(sorted(step_known))]),
            # step_max / (len(step_known) - 1),
            # statistics.mode([abs(a - b) for a, b in itertools.pairwise(step_known)])
            step_max
            - step_min
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
                S_debug = init_S_debug(search_interval, np)
                f_debug = np.array([fg(x_old + s * d)[0] for s in S_debug])
            print_debug_info(
                GP_posterior,
                S_debug,
                (f_debug + normalization_offset) * normalization_scale,
                step_known,
                f_known_normalized,
                acquisitionFunction,
            )

        k += 1

    if debug_options.plot_gp:
        if S_debug is None:
            S_debug = init_S_debug(search_interval, np)
            f_debug = np.array([fg(x_old + s * d)[0] for s in S_debug])
        print_debug_info(
            GP_posterior,
            S_debug,
            (f_debug + normalization_offset) * normalization_scale,
            step_known,
            (f_known + normalization_offset) * normalization_scale,
            acquisitionFunction,
        )

    return return_best_step(
        x_old,
        f_old,
        g_old,
        d,
        step_known,
        f_known,
        g_known,
        fun_eval,
        np,
        debug_options,
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
    np = value_or_value(np, numpy)
    debug_options = value_or_func(debug_options, LineSearchDebugOptions)

    # For computation of directional derivatives we need the unit vector of the direction. Since step is scaled by d, we add the size back in though.
    # d_norm = d / np.linalg.norm(d)
    d_norm = d

    # Test to ensure that d is a descent direction
    if dg := np.dot(g_old, d) >= 0:
        assert (
            False
        ), f"Descent direction should be descent direction. Directional gradient was {dg} in direction {d}"

    step_min = 0.0
    step_max = min(float(step_max), 1.0)  # Start initially with at most 1.0

    total_fun_eval = 0

    step = None

    k = 0

    data_points = [DataPoint(0.0, x_old, f_old, g_old, np.dot(g_old, d_norm))]

    while True:
        f, g, x, step, fun_eval, wolfe_met = gp_line_search(
            x_old,
            d,
            d_norm,
            fg,
            (step_min, step_max),
            f_old,
            g_old,
            data_points,
            np,
            debug_options,
            max_sample_count,
        )

        total_fun_eval += fun_eval

        if wolfe_met:
            if debug_options.report_wolfe_termination:
                print(f"Wolfe after {k} iterations")
            return f, g, x, step, total_fun_eval

        # Stop if any of the termination conditions are met
        if step is not None and step != step_max:
            if debug_options.report_wolfe_termination:
                print(
                    f"Terminated line search due to {'better' if f < f_old else 'equal'} value found after {k} iterations"
                )

            return f, g, x, step, total_fun_eval

        if k > max_iter:
            if debug_options.report_wolfe_termination:
                print("Terminated line search due to exceeded iteration count")
            return f, g, x, step, total_fun_eval

        if step is not None and step == step_max:
            # Double the search are, since the Bayesian optimization expects the optimum to be past step_max
            step_min = step_max
            step_max *= 2.0

            if debug_options.report_area_reduction:
                print(
                    f"Minimum seems to be past step_max. Restarting with search_interval={(step_min, step_max)}"
                )
        else:
            # Reduce the search area, since gp_line search could not find step on current search area (too big, this large instability or thorough search)
            # TODO: Reduce area to be around best value found yet maybe
            data_points.sort(key=lambda d: d.step)
            assert step_min == 0.0
            # No need to consider step_min here, since we will never decrease step_max if step_min is not 0.0
            step_max = min(
                data_points[5].step if len(data_points) > 5 else step_max,
                step_max * 0.5,
            )

            if debug_options.report_area_reduction:
                print(
                    f"Could not find step. Restarting with search_interval={(step_min, step_max)}"
                )

        k += 1
