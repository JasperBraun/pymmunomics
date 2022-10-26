from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Iterable, Sequence, Union

from numpy import arange, array, empty, isin
from numpy.typing import ArrayLike
from pandas import concat, DataFrame, Series
from scipy.stats import kendalltau
from scipy.stats.mstats import mquantiles
from sklearn.base import TransformerMixin

from pymmunomics.helper.exception import InvalidArgumentError


class GroupedTransformer(TransformerMixin):

    def __init__(
        self,
        column_ranges: Iterable[Sequence],
        transformers: Iterable[TransformerMixin],
    ):
        """Applies different transformers to runs of columns independently.

        Parameters
        ----------
        column_ranges:
            The groups of column headers to apply transformers to.
        transformers:
            The transformers correponding to the column header groups
            specified by `column_ranges`.

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
        return concat(
            [
                transformer.transform(X[column_range])
                for column_range, transformer
                in zip(self.column_ranges, self.transformers)
            ],
            axis=1,
        )

class NullScoreSelectorBase(TransformerMixin, ABC):
    def __init__(
        self,
        null_data: DataFrame,
        null_y: str,
        train_data: Union[DataFrame, None] = None,
        train_y: Union[str, None] = None,
        score_func: Callable = lambda x, y: kendalltau(x, y, variant="c")[0],
        alpha: float = 0.05,
    ):
        """Base class for variable selectors via null distribution of scores.

        Parameters
        ----------
        null_data, train_data, null_y, train_y:
            Rows correspond to samples, columns to variables. Columns
            `null_y`, and `train_y` designate the dependent variable,
            all others are independent variables. Index and column
            headers are used to identify samples and variables,
            respectively. Either both, or neither of `train_data`
            and `train_y` must be provided.
        score_func:
            Function that takes two numpy arrays and returns a score.
            Will be used to assign scores to variables in null and
            train data. Called with one column of null or train data as
            the first argument and the dependent variable column as the
            second.
        alpha:
            Determines how extreme selected variable scores have to be
            compared to null scores.
        """
        if sum(map(lambda arg: arg is None, [train_data, train_y])) == 1:
            raise InvalidArgumentError(
                "Must specify both or neither of `train_data` and `train_y`"
            )

        self.null_y = null_data.pop(null_y).to_numpy()
        self.null_X = null_data
        if train_y is not None:
            self.train_y = train_data.pop(train_y).to_numpy()
            self.train_X = train_data
        else:
            self.train_y, self.train_X = (None, None)
        self.score_func = score_func
        self.alpha = alpha

        self.null_scores = empty(self.null_X.shape[1])
        self.train_scores = None
        self.lower_quantile = None
        self.upper_quantile = None
        self.selected_columns = None

    @abstractmethod
    def fit(self, X, y):
        pass

    def transform(self, X):
        """Filters `X` returning only the selected columns in `X`."""
        return X[self.selected_columns]

    def _decide_train_Xy(self, X, y):
        if self.train_X is None:
            self.train_X, self.train_y = X, y

    def _calculate_scores(self, train_index_items):
        self.train_scores = empty(self.train_X.shape[1])
        for full_index, X, y, out in [
            (self.null_X.index, self.null_X, self.null_y, self.null_scores),
            (self.train_X.index, self.train_X, self.train_y, self.train_scores),
        ]:
            train_index = full_index.isin(train_index_items)
            for j, col in enumerate(X.to_numpy()[train_index].T):
                out[j] = self.score_func(col, y[train_index])

class SelectNullScoreOutlier(NullScoreSelectorBase):
    def fit(self, X: DataFrame, y: ArrayLike):
        """Calculates scores and determines outlying variables.

        Selected variables are those whose score is below the
        `alpha / 2`'th or above the `1 - (alpha / 2)`'th quantile of
        null scores.

        Parameters
        ----------
        X:
            The train data from which to select outlying independent
            variables. The index will identify which samples in the null
            data to use for generating the null score distribution.
        y:
            The dependent variable's values corresponding in order to
            the train data in `X`.

        Notes
        -----
        - If `train_data` and `train_y` were passed as arguments
          to the initializer, arguments `X` and `y` are ignored, except
          for determining which samples are to be used.

        Returns
        -------
        self:
            Fitted feature-selector.
        """
        self._decide_train_Xy(X, y)
        self._calculate_scores(set(X.index))
        self.lower_quantile, self.upper_quantile = mquantiles(
            self.null_scores,
            prob=[self.alpha / 2, 1 - (self.alpha / 2)],
        )
        self.selected_columns = self.train_X.columns[
            (self.train_scores < self.lower_quantile)
            | (self.upper_quantile < self.train_scores)
        ]
        # breakpoint()
        return self

class SelectPairedNullScoreOutlier(NullScoreSelectorBase):
    def fit(self, X, y):
        """Calculates scores and determines outlying variables.

        Selected variables are those whose difference in null and
        train data score is below the `alpha / 2`'th or above the
        `1 - (alpha / 2)`'th quantile of all score differences.

        Parameters
        ----------
        X:
            The train data from which to select outlying independent
            variables. The index will identify which samples in the null
            data to use for generating the null score distribution.
        y:
            The dependent variable's values corresponding in order to
            the train data in `X`.

        Notes
        -----
        - If `train_data` and `train_y` were passed as arguments
          to the initializer, arguments `X` and `y` are ignored, except
          for determining which samples are to be used.

        Returns
        -------
        self:
            Fitted feature-selector.
        """
        self._decide_train_Xy(X, y)
        self.train_X = self.train_X[self.null_X.columns]
        self._calculate_scores(set(X.index))
        delta_scores = self.train_scores - self.null_scores
        self.lower_quantile, self.upper_quantile = mquantiles(
            delta_scores,
            prob=[self.alpha / 2, 1 - (self.alpha / 2)],
        )
        self.selected_columns = self.train_X.columns[
            (delta_scores < self.lower_quantile)
            | (self.upper_quantile < delta_scores)
        ]
        return self
