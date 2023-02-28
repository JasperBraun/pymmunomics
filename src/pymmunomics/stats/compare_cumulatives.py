"""Module for comparison of cumulatives."""

def get_best_separator_pos(items: ndarray):
    """Gets position which best separates items into two groups.

    Best separating position `i` optimizes the absolute value of the sum
    of items before plus the absolute value of the sum of items after
    the position:
    ```
        abs(items[0] + ... + items[i-1]) + abs(items[i] + ... + items[n])
    ```
    where `n` is the length of `items`

    Parameters
    ----------
    items:
        1-d numeric array to find a best separating position for.

    Returns
    -------
    pos:
        Position which best separates `items` into 2 groups.
    """
    if len(items.shape) != 1:
        raise InvalidArgumentError(
            "Expect 1-d array, but got shape: %s" % items.shape
        )
    separator_scores = map(
        lambda i: abs(sum(items[:i])) + abs(sum(items[i:])),
        arange(len(items))
    )
    return argmax(separator_scores)

class CompareBestSlopeSeparatorCumulativesResult(NamedTuple):
    pvalue: float,
    cumulatives: Tuple[ndarray, ndarray],
    slopes: ndarray,
    best_separator_pos: int,

def _mannwhitneyu(a: ndarray, b: ndarray):
    return mannwhitneyu(a, b).pvalue

def _validate_compare_best_slope_separator_cumulatives(
    a: ndarray,
    b: ndarray,
    slope: Union[Callable, ndarray],
    test_func: Callable,
) -> None:
    if len(a.shape) != 2 or len(b.shape) != 2 or a.shape[1] != b.shape[1]:
        raise InvalidArgumentError(
            "Invalid shapes for a %s, or b %s. Must be 2-dimensional"
            " and have same number of columns."
            % (a.shape, b.shape)
        )
    if a.shape[0] == 0 or b.shape[0] == 0:
        raise InvalidArgumentError("One or more empty input arrays.")
    if not callable(slope) and (
        len(slope.shape) != 1 or slope.shape[0] != a.shape[1]
    ):
        raise InvalidArgumentError(
            "Invalid shape for slope %s. Must be 1-dimensional with one"
            " entry per column in a or b."
        )

def compare_best_slope_separator_cumulatives(
  a: ndarray,
  b: ndarray,
  slope: Union[Callable, ndarray],
  test_func: Callable=_mannwhitneyu,
) -> CompareBestSlopeSeparatorCumulativesResult:
    """Compares cumulatives up to best slope-separating line.

    Parameters
    ----------
    a, b:
        2-d arrays of frequencies for the two groups. One row per
        observation. Must have same number of columns.
    slope:
        If callable, should take a column from `a` and one from `b` and
        return a slope, where independent variable takes values 0 and 1
        for `a` and `b`, respectively. If numpy.ndarray, should be
        1-dimensional and have one slope per support value (per column
        in `a` or `b`).
    test_func:
        Should take two arrays of cumulatives of `a` and `b` up to the
        best separating support value and return a p-value. Uses
        two-tailed Mann-Whitney U by default.
    """
    _validate_compare_best_slope_separator_cumulatives(
        a=a, b=b, slope=slope, test_func=test_func,
    )

    if callable(slope):
        slopes = [slope(a_col, b_col) for a_col, b_col in zip(a.T, b.T)]
    else:
        slopes = slope

    best_separator_pos = get_best_separator_pos(slopes)
    cumulatives_a = a[:,:i].sum(axis=1)
    cumulatives_b = b[:,:i].sum(axis=1)
    result = CompareBestSlopeSeparatorCumulativesResult(
        pvalue=test_func(cumulatives_a, cumulatives_b),
        cumulatives=(cumulatives_a, cumulatives_b),
        slopes=slopes,
        best_separator_pos=best_separator_pos,
    )
    return result
