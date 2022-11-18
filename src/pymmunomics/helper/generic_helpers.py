# Make into class
# args kwargs as __call__ args and functions function_kwargs as init? function_kwargs also as __call__ arg?
def pipeline(
    functions: Sequence[Callable],
    *args,
    function_kwargs: Union[dict, Sequence[dict], None] = None,
    **kwargs,
):
    """Applies chain of function to initial arguments.

    Parameters
    ----------
    functions:
        The functions to apply in that order. Each function is applied
        to its predecessor's return value. The first function is applied
        to `args`, and `kwargs`. The last function's return value is
        returned.
    function_kwargs:
        If ``None``, does nothing. If ``dict``, is passed as keyword
        arguments to each of the functions. If ``Sequence[dict]``, it
        must correspond in length with `functions` and items are passed
        as keyword arguments to corresponding members of `functions`.
    args, kwargs:
        The first function's positional and keyword arguments.

    Returns
    -------
    Return value of last member of `functions`.
    """
    if function_kwargs is None:
        function_kwargs = list(repeat({}, len(functions)))
    elif type(function_kwargs) == dict:
        function_kwargs = list(repeat(function_kwargs, len(functions)))
    elif len(function_kwargs) != len(functions):
        raise InvalidArgumentError(
            "Must pass same number of function_kwargs as functions."
            " Instead, passed %s functions and %s function_kwargs",
            (len(functions), len(function_kwargs)),
        )
    result = functions[0](*args, **kwargs, **function_kwargs[0])
    for func, kwargs in zip(functions[1:], function_kwargs[1:]):
        result = func(result, **kwargs)
    return result
