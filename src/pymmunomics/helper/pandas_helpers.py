from typing import Callable, Iterable, Mapping, Optional, Sequence, Union

from pandas import concat, DataFrame


def apply_partial_pooled_grouped(
    data_frame: DataFrame,
    func: Callable,
    by: Sequence[str],
    pooled: Iterable[Iterable[str]],
    *func_args,
    pooled_val: str = "pooled",
    **func_kwargs,
) -> DataFrame:
    """Applies func to various groupings and poolings of data."""
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
