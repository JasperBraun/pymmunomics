from typing import Any, Iterable, List, Union

from pandas import concat, DataFrame


def count(
    seq: DataFrame,
    feature_groups: Iterable[Union[str, List[str]]],
    clonesize_column: str,
) -> DataFrame:
    """Grouped counts of clonotype features.

    Parameters
    ----------
    seq:
        Table of clonotypes and their clone sizes.
    feature_groups:
        Columns in seq representing different clonotype features.
    clonesize_column:
        Column in seq listing clone sizes corresponding to the
        clonotypes.

    Returns
    -------
    A table indexed by the values of feature_groups (index level
    'feature_group') and the corresponding column values (index level
    'feature') with single column 'count' that contains the sum of the
    clone sizes of clonotypes exhibiting the indexed clonotype features.

    Example
    -------

    """
    counts_list = []
    for feature_group in feature_groups:
        counts_list.append(
            seq.groupby(feature_group)[[clonesize_column]]
            .sum()
            .pipe(
                lambda df: df.assign(
                    feature=df.index.to_flat_index(),
                    feature_group=(
                        feature_group
                        if type(feature_group) == str
                        else str(feature_group)
                    ),
                )
            )
            .set_index(["feature_group", "feature"])
            .rename(columns={clonesize_column: "count"})
        )
    counts = concat(counts_list)
    return counts


def frequency(
    seq: DataFrame,
    feature_groups: Iterable[str],
    clonesize_column: str,
) -> DataFrame:
    """Grouped frequencies of clonotype features."""
    total_count = seq[clonesize_column].sum()
    counts = count(
        seq=seq,
        feature_groups=feature_groups,
        clonesize_column=clonesize_column,
    )
    frequencies = counts.assign(frequency=counts["count"] / total_count).drop(
        columns="count"
    )
    return frequencies
