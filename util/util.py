def value_or_func(value, func, *args, **kwargs):
    """A helper function that passes the value through if it is not None and evaluates the function otherwise"""
    return value if value is not None else func(*args, **kwargs)


def value_or_value(lhs, rhs):
    """A helper function that passes through the lhs value if it is not None and the rhs value otherwise"""
    return lhs if lhs is not None else rhs


# TODO: replace x = x if x is not None else y with these
