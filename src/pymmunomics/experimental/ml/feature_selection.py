from abc import ABC, abstractmethod
from functools import partial
from math import ceil
from multiprocessing import cpu_count
from typing import Any, Callable, Iterable, Mapping, Sequence, Union

from numpy import arange, array, concatenate, empty, isin
from numpy.typing import ArrayLike
from pandas import concat, DataFrame, isna, MultiIndex, Series
from ray import remote, get, put
from scipy.stats import kendalltau
from scipy.stats.mstats import mquantiles
from sklearn.base import BaseEstimator, TransformerMixin
from skopt.space.space import Dimension

from pymmunomics.helper.exception import InvalidArgumentError

class IdentityDimension(ABC):
    def __init__(self):
        self.transformer = FunctionTransformer()

    def transform(self, X):
        return self.transformer.transform(X)

    def inverse_transform(self, Xt):
        return self.transformer.inverse_transform(Xt)

    @abstractmethod
    def rvs(self, n_samples=1, random_state=None):
        pass

class DimensionList(Dimension, IdentityDimension):
    def __init__(
        self,
        dimensions: Sequence[Mapping[str, Dimension]],
    ):
        self.dimensions = dimensions

    def rvs(self, n_samples=1, random_state=None):
        sample = []
        for dimension_map in self.dimensions:
            item = {}
            for param, dimension in dimension_map.items():
                item[param] = dimension.rvs(
                    n_samples=n_samples,
                    random_state=random_state,
                )
            sample.append(item)
        return sample

    def set_transformer(self, transform="identity"):
        for i in range(len(self.dimensions)):
            for key in self.dimensions[i]:
                self.dimensions[i][key].set_transformer(transform=transform)

class GroupedTransformer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        column_ranges: Union[Sequence[Sequence], None] = None,
        transformers: Union[Sequence[TransformerMixin], None] = None,
        params: Union[Sequence[dict], None] = None,
        flatten_columns: bool = True,
    ):
        """Applies different transformers to runs of columns independently.

        Parameters
        ----------
        column_ranges:
            The groups of column headers to apply transformers to.
        transformers:
            The transformers correponding to the column header groups
            specified by `column_ranges`.
        params:
            Keyword arguments for corresponding transformers
        flatten_columns:
            Convert ``pandas.MultiIndex`` columns to strings of tuples
            during transformation.

        Notes
        -----
        - **Important:** This class expects ``pandas.DataFrame`` objects for
          its parameter `X` in `fit` and `transform`, as well as for the
          return values of the `transform` method of the individual
          transformers.
        - Each transformer (class implementing `fit` and `transform` methods;
          see ``sklearn.base.TransformerMixin``) corresponds to a range of
          columns and its `fit` and `transform` methods are called on the
          corresponding column range of the input data.
        - Implements basic version of ``sklearn.base.TransformerMixin``.
          Optional arguments in function signatures are omitted.
        - Assumes data to be in ``pandas.DataFrame`` format.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from sklearn.base import TransformerMixin
        >>> 
        >>> from pymmunomics.ml.feature_selection import GroupedTransformer
        >>> 
        >>> class AddNum(TransformerMixin):
        ...     '''Adds constant to data.'''
        ...     def __init__(self, num):
        ...         self.num = num
        ...     def fit(self, X, y):
        ...         return self
        ...     def transform(self, X):
        ...         return X + self.num
        >>> 
        >>> X = pd.DataFrame(
        ...     columns=["f1", "f2", "g1", "g2"],
        ...     data=[
        ...         [0, 0, 0, 0],
        ...         [0, 0, 0, 0],
        ...         [0, 0, 0, 0],
        ...         [0, 0, 0, 0],
        ...         [0, 0, 0, 0],
        ...     ],
        ... )
        >>> y = np.arange(X.shape[0]) # only needed due to .fit function signature
        >>> f_transformer = AddNum(1)
        >>> f_range = ["f1", "f2"]
        >>> g_transformer = AddNum(-1)
        >>> g_range = ["g1", "g2"]
        >>> transformer = GroupedTransformer(
        ...     column_ranges=[f_range, g_range],
        ...     transformers=[f_transformer, g_transformer],
        ... )
        >>> transformer.fit_transform(X, y)
           f1  f2  g1  g2
        0   1   1  -1  -1
        1   1   1  -1  -1
        2   1   1  -1  -1
        3   1   1  -1  -1
        4   1   1  -1  -1
        """
        self.column_ranges = column_ranges
        self.transformers = transformers
        self.params = params
        self.flatten_columns = flatten_columns

    def fit(self, X: DataFrame, y: Any):
        """Fits transformers on their respective columns of X.

        Parameters
        ----------
        X:
            Indexed by each of the column ranges and passed to the
            corresponding transformer's `.fit` method.
        y:
            Passed to the `.fit` method of all transformers

        Returns
        -------
        self:
            Fitted transformer.
        """
        if self.column_ranges is None:
            self.columns_ranges = [X.columns]
        if self.transformers is None:
            self.transformers = [FunctionTransformer()]
        if self.params is None:
            self.params = [{} for _ in range(len(self.transformers))]
        for transformer, params_ in zip(self.transformers, self.params):
            transformer.set_params(**params_)
        for column_range, transformer in zip(self.column_ranges, self.transformers):
            transformer.fit(X[column_range], y)
        return self

    def transform(self, X: DataFrame):
        """Applies transformers to their respective column ranges in X.

        Parameters
        ----------
        X:
            Indexed by each of the column ranges and passed to the
            corresponding transformer's `.transform` method.

        Returns
        -------
        X_new:
            The transformed data.
        """
        result = concat(
            [
                transformer.transform(X[column_range])
                for column_range, transformer
                in zip(self.column_ranges, self.transformers)
            ],
            axis=1,
        )
        if self.flatten_columns and type(result.columns) == MultiIndex:
            flat_names = str(tuple(result.columns.names))
            flat_columns = result.columns.to_flat_index().map(str)
            result.columns = flat_columns
            result.columns.name = flat_names
        return result