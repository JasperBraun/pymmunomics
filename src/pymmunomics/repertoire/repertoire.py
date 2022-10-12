from typing import Iterable, Mapping, Sequence, Union

from pandas import DataFrame

from pymmunomics.helper.pandas_helpers import (
    apply_partial_pooled_grouped,
    pivot_pipe_melt,
)
from pymmunomics.repertoire.clonotype import (
    count,
    frequency,
)

# def group_features(
#     feature_table: DataFrame,
#     # repertoire_groups: Sequence[str],
#     combine_repertoire_groups: Mapping[str, Iterable[Sequence[str]]],
#     prepend_group: Union[Sequence[str], None],
#     inplace: bool = False
# ):
#     if inplace:
#         grouped_features = feature_table
#     else:
#         grouped_features = feature_table.copy()
#     if prepend_group is None:
#         prepend_group = []

#     for combine_col, combine_vals in combine_repertoire_groups.items():
#         constant_cols = [
#             col for col in grouped_features.columns
#             if col != combine_col
#             and col not in prepend_group
#         ]
#         all_combine_cols = []
#         for vals in combine_vals:
#             for val in vals:
#                 all_combine_cols.append(val)
#         # figure out prepend stuff
#             all_combine_cols.col for col in vals for vals in combine_vals]
#         for vals in combine_vals:
#             if len(vals) > 1:
#                 for group_col in prepend_group:
#                     grouped_features[group_col] = [
#                         "_".join([group_val, prepend_val])
#                         if group_val in vals
#                         for group_val, prepend_val
#                         in zip(grouped_features[group_col], grouped_features[prepend_val])
#                     ]
#                 combined = "_".join(vals)
#                 grouped_features[column] = grouped_features[column].map(
#                     lambda val: combined if val in vals else val
#                 )


def count_clonotype_features(
    repertoire: DataFrame,
    repertoire_groups: Sequence[str],
    clonotype_features: Sequence[str],
    clonesize: str,
    partial_repertoire_pools: Union[Iterable[Sequence[str]], None] = None,
    relative_counts: bool = True,
    shared_clonotype_feature_groups: Union[Sequence[str], None] = None,

):
    """Counts various clonotype feature occurrences in repertoires.

    Parameters
    ----------
    repertoire:
        Table of clonotypes and their clone sizes as well as columns
        indicating repertoire membership.
    repertoire_groups:
        Columns indicating repertoire memberships.
    clonotype_features:
        Columns representing different clonotype features.
    clonesize:
        Column listing the clone sizes in corresponding to the
        clonotypes and repertoires.
    partial_repertoire_pools:
        If specified, counts are calculated and concatenated once for
        each member of this list with the indicated columns' values
        replaced by the value "pooled" effectively combining repertoires
        with same values in these columns.
    relative_counts:
        Whether or not to provide frequency counts as opposed to
        absolute counts.
    shared_clonotype_feature_groups:
        If provided, all repertoires with the same value in these
        columns are forced to have frequencies for the same clonotype
        feature values per feature group. Missing counts are set to 0.

    Returns
    -------
    feature_counts:
        Table listing frequency per repertoire, feature and feature
        value. Columns are: `repertoire_groups` which indicate
        repertoire membership of a count, "feature" which contains one
        of `clonotype_features` indicating which clonotype feature the
        row counts, "feature_value" indicating a count's corresponding
        counted value of the feature and finally, "frequency", or
        "count", depending on the value of `relative_counts`, which
        contain the actual counts.

    Examples
    --------
    >>> import pandas as pd
    >>> from pymmunomics.repertoire.repertoire import count_clonotype_features
    >>> 
    >>> repertoire = pd.DataFrame(
    ...     columns=[
    ...         "g1", "g2", "f1", "f2", "clonesize",
    ...     ],
    ...     data=[
    ...         ["a", "a", 1, "x", 1],
    ...         ["a", "b", 2, "x", 2],
    ...         ["a", "a", 3, "y", 10],
    ...         ["b", "a", 1, "x", 20],
    ...         ["a", "b", 2, "y", 100],
    ...         ["a", "b", 3, "x", 200],
    ...         ["a", "a", 1, "y", 1000],
    ...     ],
    ... )
    >>> count_clonotype_features(
    ...     repertoire=repertoire,
    ...     repertoire_groups=["g1", "g2"],
    ...     clonotype_features=["f1", "f2"],
    ...     clonesize="clonesize",
    ...     partial_repertoire_pools=[[], ["g1"]],
    ...     shared_clonotype_feature_groups=["g1"],
    ... )
            g1 g2 feature feature_value  frequency
    0        a  a      f1             1   0.990109
    1        a  a      f1             3   0.009891
    2        a  a      f2             x   0.000989
    3        a  a      f2             y   0.999011
    4        a  b      f1             2   0.337748
    5        a  b      f1             3   0.662252
    6        a  b      f2             x   0.668874
    7        a  b      f2             y   0.331126
    8        b  a      f1             1   1.000000
    9        b  a      f1             3   0.000000
    10       b  a      f2             x   1.000000
    11       b  a      f2             y   0.000000
    12  pooled  a      f1             1   0.990301
    13  pooled  a      f1             3   0.009699
    14  pooled  a      f2             x   0.020369
    15  pooled  a      f2             y   0.979631
    16  pooled  b      f1             2   0.337748
    17  pooled  b      f1             3   0.662252
    18  pooled  b      f2             x   0.668874
    19  pooled  b      f2             y   0.331126
    """
    if partial_repertoire_pools is None:
        partial_repertoire_pools = [[]]
    if relative_counts:
        count_func = frequency
        count_col = "frequency"
    else:
        count_func = count
        count_col = "count"
    counts = (
        apply_partial_pooled_grouped(
            data_frame=repertoire,
            func=count_func,
            by=repertoire_groups,
            pooled=partial_repertoire_pools,
            features=clonotype_features,
            clonesize=clonesize,
        )
        .reset_index()
        .astype({"value": str})
        .rename(columns={
            "value": "feature_value",
        })
    )
    
    if shared_clonotype_feature_groups is not None:
        groupby_cols = [
            "feature",
            *[
                col for col in repertoire_groups
                if col not in shared_clonotype_feature_groups
            ]
        ]
        repertoire_feature_table = (
            counts
            .groupby(groupby_cols)
            .apply(
                pivot_pipe_melt,
                DataFrame.fillna,
                values=[count_col],
                columns="feature_value",
                index=shared_clonotype_feature_groups,
                value=0,
            )
            .reset_index()
            [[
                *repertoire_groups,
                "feature",
                "feature_value",
                count_col,
            ]]
        )
    else:
        repertoire_feature_table = counts

    if not relative_counts:
        repertoire_feature_table["count"] = repertoire_feature_table["count"].astype(int)

    return (
        repertoire_feature_table
        .sort_values([*repertoire_groups, "feature", "feature_value"])
        .reset_index(drop=True)
    )



