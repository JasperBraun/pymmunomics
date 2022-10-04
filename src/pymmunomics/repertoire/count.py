from typing import Any, Iterable, List, Union

from pandas import concat, DataFrame


def count(
    seq: DataFrame,
    feature_groups: Iterable[List[str]],
    clonesize_column: str,
) -> DataFrame:
    """Grouped counts of clonotype features."""
    counts_list = []
    for feature_group in feature_groups:
        counts_list.append(
            seq.groupby(feature_group)[[clonesize_column]]
            .sum()
            .pipe(
                lambda df: df.assign(
                    feature=df.index.to_flat_index(),
                    feature_group=str(feature_group),
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


def repertoire_size():
    pass
