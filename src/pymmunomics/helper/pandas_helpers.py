from typing import Callable, Iterable, Mapping, Optional, Sequence, Union

from pandas import concat, DataFrame


def apply_partial_pooled_grouped(
    data_frame: DataFrame,
    func: Callable,
    by: Sequence[str],
    pooled: Iterable[Sequence[str]],
    *func_args,
    pooled_val: str = "pooled",
    **func_kwargs,
) -> DataFrame:
    """Applies func to partial groupings data.

    Parameters
    ----------
    data_frame:
        Data to group and apply function to.
    func:
        Must take a group of `data_frame` grouped by `by` and return a
        ```DataFrame``. Additional arguments and keyword argument can be
        passed using `*func_args` and `*func_kwargs`.
    by:
        The columns to group by if no columns were pooled.
    pooled:
        The various subsets of `by` whose subgroups in the grouped
        `data_frame` should be pooled for the different applications of
        `func` on the resulting partially pooled groups.
    pooled_val:
        The value to assign to the columns defined by members of
        `pooled` whenever their subgroups are pooled.
    *func_args, **func_kwargs:
        Additional positional and keyword arguments to forward to func
        in addition to each data group.

    Returns
    -------
    func_results:
        Concatenation of results of func applied to the various data
        groupings. Index lists the values for columns listed in `by`
        indicating the group func was applied to. Pooled columns will
        have `pooled_val` in this index.

    Examples
    --------
    >>> import pandas as pd
    >>> from pymmunomics.helper.pandas_helpers import apply_partial_pooled_grouped
    >>> 
    >>> data_frame = pd.DataFrame(
    ...     columns=["group1", "group2", "val"],
    ...     data=[
    ...         ["a", "a", 1],
    ...         ["a", "a", 2],
    ...         ["b", "a", 3],
    ...         ["b", "a", 5],
    ...         ["a", "b", 7],
    ...         ["a", "b", 11],
    ...         ["b", "a", 13],
    ...     ],
    ... )
    >>> apply_partial_pooled_grouped(
    ...     data_frame=data_frame,
    ...     func=lambda df: df[["val"]].prod(),
    ...     by=["group1", "group2"],
    ...     pooled=[[], ["group2"], ["group1", "group2"]],
    ... )
                     val
    group1 group2       
    a      a           2
           b          77
    b      a         195
    a      pooled    154
    b      pooled    195
    pooled pooled  30030
    """
    return concat(
        [
            (
                (
                    data_frame.drop(columns=pcols)
                    .assign(**{col: pooled_val for col in pcols})
                    .groupby(by)
                ).apply(func, *func_args, **func_kwargs)
            )
            for pcols in pooled
        ],
    )


def pivot_pipe_melt(
    data_frame: DataFrame,
    func: Callable,
    values: Sequence[str],
    columns: Union[str, Sequence[str]],
    *func_args,
    index: Optional[Union[str, Sequence[str]]] = None,
    **func_kwargs,
) -> DataFrame:
    """Pipes pivotted data into func and melts result.

    Parameters
    ----------
    data_frame:
        Data to pivot, pipe and melt.
    func:
        Must take pivotted `data_frame` with columns partially selected
        for one of `values` at a time as input and return a
        ``DataFrame``.
    values, columns, index:
        Used as argument for same-named parameters in
        ``pandas.DataFrame.pivot``. If single value column, then a
        sequence of length 1 (ex.: ["value_col"]) must be used.
    *func_args, **func_kwargs:
        Additional positional and keyword arguments to forward to func
        in addition to the pivotted data.

    Returns
    -------
    func_results:
        Concatenation (along axis=1) of melted return values of `func`
        applied to pivotted data with partially selected columns for
        members of `values` one at a time. Index consists of
        combinations of values of `columns` and either values of
        original index of `data_frame`, or values in `index` columns if
        provided.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pymmunomics.helper.pandas_helpers import pivot_pipe_melt
    >>> 
    >>> data_frame = pd.DataFrame(
    ...     columns=["idx", "col", "val1", "val2"],
    ...     data=[
    ...         ["a", "a", 1, 1],
    ...         ["a", "b", 2, np.nan],
    ...         ["a", "c", 3, 3],
    ...         ["b", "a", 4, 4],
    ...         ["b", "b", 5, np.nan],
    ...     ],
    ... )
    >>> data_frame
      idx col  val1  val2
    0   a   a     1   1.0
    1   a   b     2   NaN
    2   a   c     3   3.0
    3   b   a     4   4.0
    4   b   b     5   NaN
    >>> pivot_pipe_melt(
    ...     data_frame=data_frame,
    ...     func=pd.DataFrame.fillna,
    ...     values=["val1", "val2"],
    ...     columns="col",
    ...     index="idx",
    ...     value=0,
    ... )
             val1  val2
    idx col            
    a   a     1.0   1.0
        b     2.0   0.0
        c     3.0   3.0
    b   a     4.0   4.0
        b     5.0   0.0
        c     0.0   0.0
    """
    pivotted = data_frame.pivot(index=index, columns=columns, values=values)
    piped = func(pivotted, *func_args, **func_kwargs)
    melted = concat(
        [
            piped[val]
            .melt(value_name=val, ignore_index=(index is None))
            .set_index(columns, append=(index is not None))
            for val in values
        ],
        axis=1,
    )
    return melted.sort_index()
