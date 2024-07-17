def plot_sample(ax, X, sample, label: str) -> None:
    ax.plot(X, sample, label=label)


def plot_samples(ax, X, samples) -> None:
    for i, v in enumerate(samples):
        plot_sample(ax, X, v, f"Sample {i}")

def plot_observations(ax, X_train, y_train):
    ax.scatter(X_train, y_train, label="Observations")

def plot_objective(ax, X, y, X_train, y_train, g=None) -> None:
    ax.plot(X, y, label="Objective", linestyle="dotted")
    plot_observations(ax, X_train, y_train)
    if g is not None:
        ax.plot(X, g, label="Obj Gradient", linestyle="dotted")

def plot_gp(
    ax, X, meanValues, stdValues, label="mean", stdFactor=2.0, drawStd=True
) -> None:
    ax.plot(X, meanValues, label=label)
    if drawStd:
        ax.fill_between(
            X,
            meanValues - stdFactor * stdValues,
            meanValues + stdFactor * stdValues,
            alpha=0.25,
            label=f"{stdFactor}-std-div",
        )


def plot_label(ax, title: str) -> None:
    ax.legend()
    ax.grid()
    ax.set_title(title)
    ax.set(xlabel="$x$", ylabel="$f(x)$")
    ax.label_outer()
