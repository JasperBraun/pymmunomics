from typing import Any, Iterable, List, Union

from pandas import concat, DataFrame


def count(
    repertoire: DataFrame,
    features: Iterable[Union[str, List[str]]],
    clonesize: str,
) -> DataFrame:
    """Counts number of occurrences of clonotype features in repertoire.

    Parameters
    ----------
    repertoire:
        Table of clonotypes and their clone sizes.
    features:
        Columns in repertoire representing different clonotype features.
    clonesize:
        Column in repertoire listing clone sizes corresponding to the
        clonotypes.

    Returns
    -------
    feature_counts:
        A table indexed by features (index level 'feature') and their
        corresponding values (index level 'value') with single column
        'count' that contains the sum of the clone sizes of clonotypes
        exhibiting the indexed clonotype feature value.

    Examples
    --------
    >>> import pandas as pd
    >>> from pymmunomics.repertoire.clonotype import count
    >>> 
    >>> repertoire = pd.DataFrame(
    ...     columns=["feature_1", "feature_2", "clonesize"],
    ...     data=[
    ...         ["a", "x", 1],
    ...         ["a", "x", 10],
    ...         ["b", "x", 100],
    ...         ["b", "y", 1000],
    ...         ["c", "y", 10000],
    ...     ],
    ... )
    >>> count(
    ...     repertoire=repertoire,
    ...     features=["feature_2", ["feature_1", "feature_2"]],
    ...     clonesize="clonesize",
    ... )
                                   count
    feature                value        
    feature_2              x         111
                           y       11000
    (feature_1, feature_2) (a, x)     11
                           (b, x)    100
                           (b, y)   1000
                           (c, y)  10000
    """
    counts_list = []
    for feature in features:
        counts_list.append(
            repertoire.groupby(feature)[[clonesize]]
            .sum()
            .pipe(
                lambda df: df.assign(
                    value=df.index.to_flat_index(),
                    feature=(
                        [feature if type(feature) == str else tuple(feature)] * len(df)
                    ),
                )
            )
            .set_index(["feature", "value"])
            .rename(columns={clonesize: "count"})
        )
    counts = concat(counts_list)
    return counts


def frequency(
    repertoire: DataFrame,
    features: Iterable[str],
    clonesize: str,
) -> DataFrame:
    """Grouped frequencies of clonotype features.

    Parameters
    ----------
    repertoire:
        Table of clonotypes and their clone sizes.
    features:
        Columns in repertoire representing different clonotype features.
    clonesize:
        Column in repertoire listing clone sizes corresponding to the
        clonotypes.

    Returns
    -------
    feature_frequencies:
        A table indexed by features (index level 'feature') and their
        corresponding values (index level 'value') with single column
        'frequency' that contains the sum of the clone sizes of
        clonotypes exhibiting the indexed clonotype feature value
        normalized by the sum of all clone sizes.

    Examples
    --------
    >>> import pandas as pd
    >>> from pymmunomics.repertoire.clonotype import frequency
    >>> 
    >>> repertoire = pd.DataFrame(
    ...     columns=["feature_1", "feature_2", "clonesize"],
    ...     data=[
    ...         ["a", "x", 1],
    ...         ["a", "x", 10],
    ...         ["b", "x", 100],
    ...         ["b", "y", 1000],
    ...         ["c", "y", 10000],
    ...     ],
    ... )
    >>> frequency(
    ...     repertoire=repertoire,
    ...     features=["feature_2", ["feature_1", "feature_2"]],
    ...     clonesize="clonesize",
    ... )
                                   frequency
    feature                value            
    feature_2              x        0.009990
                           y        0.990010
    (feature_1, feature_2) (a, x)   0.000990
                           (b, x)   0.009000
                           (b, y)   0.090001
                           (c, y)   0.900009
    """
    total_count = repertoire[clonesize].sum()
    counts = count(
        repertoire=repertoire,
        features=features,
        clonesize=clonesize,
    )
    frequencies = counts.assign(frequency=counts["count"] / total_count).drop(
        columns="count"
    )
    return frequencies
