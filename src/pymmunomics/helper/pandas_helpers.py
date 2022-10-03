from typing import Callable, Iterable, List

from pandas import concat, DataFrame


def apply_partial_pooled_grouped(
    data_frame: DataFrame,
    func: Callable,
    by: List[str],
    pooled: Iterable[Iterable[str]],
    pooled_val: str = "pooled",
) -> DataFrame:
    """Applies func to various groupings and poolings of data."""
    return concat(
        [
            (
                (
                    data_frame.drop(columns=pcols)
                    .assign(**{col: pooled_val for col in pcols})
                    .groupby(by)
                ).apply(func)
            )
            for pcols in pooled
        ]
    )
