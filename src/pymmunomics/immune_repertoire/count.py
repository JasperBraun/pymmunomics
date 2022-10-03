from typing import Iterable

from pandas import concat, DataFrame


def count(
    seq: DataFrame,
    feature_groups: Iterable[str],
    clonesize_column: str,
) -> DataFrame:
    """Grouped counts of clonotype features."""
    counts_list = []
    for feature_group in feature_groups:
        counts_list.append(
            seq.groupby(feature_group)[[clonesize_column]]
            .sum()
            .reset_index()
            .rename(
                columns={feature_group: "feature", clonesize_column: "count"},
            )
            .assign(feature_group=feature_group)[["feature_group", "feature", "count"]][
                ["feature_group", "feature", "count"]
            ]
        )
    counts = concat(
        counts_list,
        ignore_index=True,
    )
    return counts


def frequency(
    seq: DataFrame,
    feature_groups: Iterable[str],
    clonesize_column: str,
) -> DataFrame:
    """Grouped frequencies of clonotype features."""
    total_counts = seq[clonesize_column].sum()
    counts = count(
        seq=seq,
        feature_groups=feature_groups,
        clonesize_column=clonesize_column,
    )
    frequencies = counts.assign(frequency=counts["count"] / total_counts).drop(
        columns="count"
    )
    return frequencies


def repertoire_size():
    pass
