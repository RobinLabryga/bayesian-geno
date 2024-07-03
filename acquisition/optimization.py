from acquisition import AcquisitionFunction
import itertools
import numpy as np  # Use member
import bisect
import scipy.optimize

# TODO: Comments and types


class AcquisitionOptimizer:
    def maximize(
        self,
        acquisition: AcquisitionFunction,
        lower_bound: float,
        upper_bound: float,
        x_known,
    ) -> float:
        assert False, "Not implemented"


class ScipyAcquisitionOptimizer(AcquisitionOptimizer):
    def maximize(
        self,
        acquisition: AcquisitionFunction,
        lower_bound: float,
        upper_bound: float,
        x_known,
    ) -> float:
        # We minimize the negative to maximize the acquisition.
        nAcquisitionF = lambda x: -acquisition(x)
        nAcquisitionG = lambda x: -acquisition.derivative(x)
        # TODO: Add methods to everything to combine evaluations of value and gradient (.fg())

        step = lower_bound
        delta_s = 1e-15
        starting_points = {
            s
            for a, b in itertools.pairwise(sorted(x_known))
            for s in [
                a + delta_s,
                3 * a / 4 + b / 4,
                (a + b) / 2.0,
                a / 4 + 3 * b / 4,
                b - delta_s,
            ]
            if s >= lower_bound and s <= upper_bound
        }
        for s in starting_points:
            result = scipy.optimize.minimize(
                nAcquisitionF,
                s,
                jac=nAcquisitionG,
                bounds=[(lower_bound, upper_bound)],
                method="L-BFGS-B",
            )
            step = result.x if acquisition(result.x) > acquisition(step) else step
        return step


class GradientBinarySearchAcquisitionOptimizer(AcquisitionOptimizer):
    def maximize(
        self,
        acquisition: AcquisitionFunction,
        lower_bound: float,
        upper_bound: float,
        x_known,
    ) -> float:
        max_iterations = 10  # TODO: Make parameter of some sorts

        step = lower_bound
        step_acquisition = acquisition(step)
        for lb, ub in itertools.pairwise(sorted(x_known)):
            for t in range(max_iterations):
                half_way = (lb + ub) / 2.0
                half_way_g = acquisition.derivative(half_way)

                # Found a minima or maxima at half_way
                if half_way_g == 0.0:
                    print("encountered g of 0 at half way")
                    lb, ub = half_way, half_way
                    break  # TODO: Implement this differently

                # Search in half that gradient points towards
                elif half_way_g > 0.0:
                    lb = half_way
                else:
                    ub = half_way

            if (lb_acquisition := acquisition(lb)) > step_acquisition:
                step = lb
                step_acquisition = lb_acquisition

            if (ub_acquisition := acquisition(ub)) > step_acquisition:
                step = ub
                step_acquisition = ub_acquisition

        return step


class DIRECTAcquisitionOptimizer(AcquisitionOptimizer):
    """The DIRECT algorithm from "Lipschitzian Optimization Without the Lipschitz Constant" by Jones et. al. 1993 and revisited in "The DIRECT algorithm: 25 years Later" by Jones and Martins 2019"""

    class HyperRectangle:
        def __init__(self, center: float, size: float, value: float) -> None:
            self.center = center
            self.point = np.array((size, value))

        @property
        def size(self):
            return self.point[0]

        @size.setter
        def size(self, size):
            self.point[0] = size

        @property
        def value(self):
            return self.point[1]

        def __str__(self) -> str:
            return f"({self.center}, {self.size}, {self.value})"

        def __repr__(self) -> str:
            return str(self)

        def subdivide(self, acquisition):
            new_size = self.size / 3

            return (
                DIRECTAcquisitionOptimizer.HyperRectangle(
                    center := self.center - new_size,
                    new_size,
                    float(acquisition(center)),
                ),
                DIRECTAcquisitionOptimizer.HyperRectangle(
                    center := self.center,
                    new_size,
                    self.value,
                ),
                DIRECTAcquisitionOptimizer.HyperRectangle(
                    center := self.center + new_size,
                    new_size,
                    float(acquisition(center)),
                ),
            )

    def __init__(
        self, max_iterations: int = 20, desired_accuracy: float = 1e-4
    ) -> None:
        super().__init__()
        # TODO: Make parameters
        self.max_iterations = max_iterations
        self.desired_accuracy = desired_accuracy

    def _point_below_line(self, origin, end, point):
        angle = np.cross(end.point - origin.point, point.point - origin.point)
        if angle == 0.0:  # Collinear
            return np.linalg.norm(origin.point - point.point, 2) < np.linalg.norm(
                origin.point - end.point, 2
            )

        return angle < 0

    def _lower_right_convex_hull(self, hyperrectangles, anchor):
        """Via Jarvis march with additional anchor"""
        point_on_hull = anchor
        hull = []
        while True:
            hull.append(point_on_hull)
            end_point = point_on_hull
            for j in range(len(hyperrectangles)):
                point = hyperrectangles[j]
                if (
                    point.size < point_on_hull.size
                ):  # point is to the left of the last found point of the convex hull and can not be part of the lower right covex hull
                    continue
                if end_point == point_on_hull or self._point_below_line(
                    point_on_hull, end_point, point
                ):
                    end_point = point
            if end_point.size == point_on_hull.size:  # No movement to right
                break
            point_on_hull = end_point

        return hull

    def _identify_potentially_optimal_hyperrectangles(self, hyperrectangles, f_min):
        anchor = self.HyperRectangle(
            0.0, 0.0, f_min - self.desired_accuracy * np.abs(f_min)
        )
        # Filter hyperrectangles that are not the smallest of their size and can thus not be part of the lower right convex hull
        filtered = list(
            [
                (group := list(g))[np.argmin([h.value for h in group])]
                for k, g in itertools.groupby(hyperrectangles, key=lambda h: h.size)
            ]
        )
        hull = self._lower_right_convex_hull(filtered, anchor)
        hull.remove(anchor)
        return hull

    def maximize(
        self,
        acquisition: AcquisitionFunction,
        lower_bound: float,
        upper_bound: float,
        x_known,
    ) -> float:
        # We minimize the negative to maximize the acquisition.
        nAcquisitionF = lambda x: -acquisition(x)

        # TODO: Normalization optional?
        hyperrectangles_sorted_by_value = [
            DIRECTAcquisitionOptimizer.HyperRectangle(
                hypercube_center := (lower_bound + upper_bound) / 2.0,
                np.abs(lower_bound - upper_bound),
                float(nAcquisitionF(hypercube_center)),
            )
        ]
        hyperrectangles_sorted_by_size = [hyperrectangles_sorted_by_value[0]]
        for iteration in range(self.max_iterations):
            potentially_optimal_hyperrectangles = (
                self._identify_potentially_optimal_hyperrectangles(
                    hyperrectangles_sorted_by_size,
                    hyperrectangles_sorted_by_value[0].value,
                )
            )
            for hyperrectangle in potentially_optimal_hyperrectangles:
                left_subrectangle, center_subrectangle, right_subrectangle = (
                    hyperrectangle.subdivide(nAcquisitionF)
                )
                hyperrectangles_sorted_by_size.remove(hyperrectangle)
                hyperrectangles_sorted_by_value.remove(hyperrectangle)
                bisect.insort(
                    hyperrectangles_sorted_by_value,
                    left_subrectangle,
                    key=lambda h: h.value,
                )
                bisect.insort(
                    hyperrectangles_sorted_by_value,
                    center_subrectangle,
                    key=lambda h: h.value,
                )
                bisect.insort(
                    hyperrectangles_sorted_by_value,
                    right_subrectangle,
                    key=lambda h: h.value,
                )
                bisect.insort(
                    hyperrectangles_sorted_by_size,
                    left_subrectangle,
                    key=lambda h: h.size,
                )
                bisect.insort(
                    hyperrectangles_sorted_by_size,
                    center_subrectangle,
                    key=lambda h: h.size,
                )
                bisect.insort(
                    hyperrectangles_sorted_by_size,
                    right_subrectangle,
                    key=lambda h: h.size,
                )

        return hyperrectangles_sorted_by_value[0].center


class GlobalLocalAcquisitionOptimizer(AcquisitionOptimizer):
    """Jones and Martin suggest in "The DIRECT algorithm: 25 years Later" (also many others) to refine the result of global optimization via local optimization. This class is a very simple variant that uses scipy.optimize methods."""

    def __init__(
        self, max_iterations: int = 30, desired_accuracy: float = 1e-5
    ) -> None:
        super().__init__()

        self.desired_accuracy = desired_accuracy
        self.max_iterations = max_iterations

    def maximize(
        self,
        acquisition: AcquisitionFunction,
        lower_bound: float,
        upper_bound: float,
        x_known,
    ) -> float:
        # We minimize the negative to maximize the acquisition.
        nAcquisitionF = lambda x: -acquisition(x)
        nAcquisitionG = lambda x: -acquisition.derivative(x)

        def nAcquisitionFG(x):
            f, g = acquisition.value_derivative(x)
            return -f, -g

        bounds = [(lower_bound, upper_bound)]

        global_result = scipy.optimize.differential_evolution(
            nAcquisitionF,
            bounds,
            maxiter=self.max_iterations,
            tol=self.desired_accuracy,
        )

        try:
            local_result = scipy.optimize.minimize(
                nAcquisitionFG,
                global_result.x,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                tol=self.desired_accuracy,
                options={"maxiter": self.max_iterations},
            )
        except ValueError:
            # scipy-stubs/optimize/_lbfgsb.pyi _lbfgsb.setulb sometimes sets x to nan. This results in a ValueError that we catch here
            # TODO: Maybe look more into. Could be because gradient at x0=0.0 points to left
            print(f"Search from {global_result.x} resulted in Value error")
            return global_result.x.item()

        return local_result.x.item()
