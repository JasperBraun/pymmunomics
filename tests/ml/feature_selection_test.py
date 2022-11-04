from unittest.mock import MagicMock

from numpy import array_equal, zeros
from pandas import DataFrame, Index, MultiIndex
from pandas.testing import assert_frame_equal
from pytest import fixture
from sklearn.base import TransformerMixin

from pymmunomics.ml.feature_selection import (
    GroupedTransformer,
    SelectNullScoreOutlier,
    SelectPairedNullScoreOutlier,
)

class TestGroupedTransformer:
    @fixture
    def add_one(self):
        transformer = MagicMock()
        transformer.transform = MagicMock(side_effect=lambda X: X+1)
        return transformer

    @fixture
    def subtract_one(self):
        transformer = MagicMock()
        transformer.transform = MagicMock(side_effect=lambda X: X-1)
        return transformer

    def test_split_half_fit(self, add_one, subtract_one):
        X = DataFrame(
            columns=["f1", "f2", "g1", "g2"],
            data=zeros(shape=(5, 4), dtype=int),
        )
        y = zeros(5)
        column_ranges = [["f1", "f2"], ["g1", "g2"]]
        transformers = [add_one, subtract_one]
        
        grouped_transformer = GroupedTransformer(
            column_ranges=column_ranges,
            transformers=transformers,
        )        
        grouped_transformer.fit(X, y)
        f_X, f_y = grouped_transformer.transformers[0].fit.call_args_list[-1].args
        assert_frame_equal(f_X, X[["f1", "f2"]])
        assert array_equal(f_y, y)
        g_X, g_y = grouped_transformer.transformers[1].fit.call_args_list[-1].args
        assert_frame_equal(g_X, X[["g1", "g2"]])
        assert array_equal(g_y, y)

    def test_split_half_transform(self, add_one, subtract_one):
        X = DataFrame(
            columns=["f1", "f2", "g1", "g2"],
            data=zeros(shape=(5, 4), dtype=int),
        )
        y = zeros(5)
        column_ranges = [["f1", "f2"], ["g1", "g2"]]
        transformers = [add_one, subtract_one]
        
        grouped_transformer = GroupedTransformer(
            column_ranges=column_ranges,
            transformers=transformers,
        )        
        grouped_transformer.fit(X, y)
        expected = DataFrame(
            columns=["f1", "f2", "g1", "g2"],
            data=[
                [1, 1, -1, -1],
                [1, 1, -1, -1],
                [1, 1, -1, -1],
                [1, 1, -1, -1],
                [1, 1, -1, -1],
            ]
        )
        X_new = grouped_transformer.transform(X)
        assert_frame_equal(X_new, expected)

    def test_single_fit(self, add_one):
        X = DataFrame(
            columns=["f1", "f2", "g1", "g2"],
            data=zeros(shape=(5, 4), dtype=int),
        )
        y = zeros(5)
        column_ranges = [["f1", "f2", "g1", "g2"]]
        transformers = [add_one]
        
        grouped_transformer = GroupedTransformer(
            column_ranges=column_ranges,
            transformers=transformers,
        )        
        grouped_transformer.fit(X, y)
        fit_X, fit_y = grouped_transformer.transformers[0].fit.call_args_list[-1].args
        assert_frame_equal(fit_X, X)
        assert array_equal(fit_y, y)

    def test_single_transform(self, add_one):
        X = DataFrame(
            columns=["f1", "f2", "g1", "g2"],
            data=zeros(shape=(5, 4), dtype=int),
        )
        y = zeros(5)
        column_ranges = [["f1", "f2", "g1", "g2"]]
        transformers = [add_one]
        
        grouped_transformer = GroupedTransformer(
            column_ranges=column_ranges,
            transformers=transformers,
        )        
        grouped_transformer.fit(X, y)
        expected = DataFrame(
            columns=["f1", "f2", "g1", "g2"],
            data=[
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]
        )
        X_new = grouped_transformer.transform(X)
        assert_frame_equal(X_new, expected)

    def test_subset_fit(self, add_one, subtract_one):
        X = DataFrame(
            columns=["f1", "f2", "g1", "g2"],
            data=zeros(shape=(5, 4), dtype=int),
        )
        y = zeros(5)
        column_ranges = [["f1"], ["g1", "g2"]]
        transformers = [add_one, subtract_one]
        
        grouped_transformer = GroupedTransformer(
            column_ranges=column_ranges,
            transformers=transformers,
        )        
        grouped_transformer.fit(X, y)
        f_X, f_y = grouped_transformer.transformers[0].fit.call_args_list[-1].args
        assert_frame_equal(f_X, X[["f1"]])
        assert array_equal(f_y, y)
        g_X, g_y = grouped_transformer.transformers[1].fit.call_args_list[-1].args
        assert_frame_equal(g_X, X[["g1", "g2"]])
        assert array_equal(g_y, y)

    def test_subset_transform(self, add_one, subtract_one):
        X = DataFrame(
            columns=["f1", "f2", "g1", "g2"],
            data=zeros(shape=(5, 4), dtype=int),
        )
        y = zeros(5)
        column_ranges = [["f1"], ["g1", "g2"]]
        transformers = [add_one, subtract_one]
        
        grouped_transformer = GroupedTransformer(
            column_ranges=column_ranges,
            transformers=transformers,
        )        
        grouped_transformer.fit(X, y)
        expected = DataFrame(
            columns=["f1", "g1", "g2"],
            data=[
                [1, -1, -1],
                [1, -1, -1],
                [1, -1, -1],
                [1, -1, -1],
                [1, -1, -1],
            ]
        )
        X_new = grouped_transformer.transform(X)
        assert_frame_equal(X_new, expected)

    def test_multiindex_columns_subset_fit(self, add_one, subtract_one):
        X = DataFrame(
            columns=MultiIndex.from_arrays(
                [
                    ["a1", "a1", "a1", "a2"],
                    ["f1", "f2", "g1", "g2"],
                ],
                names=["f", "a"],
            ),
            data=zeros(shape=(5, 4), dtype=int),
        )
        y = zeros(5)
        column_ranges = [[("a1", "f1")], [("a1", "g1"), ("a2", "g2")]]
        transformers = [add_one, subtract_one]
        
        grouped_transformer = GroupedTransformer(
            column_ranges=column_ranges,
            transformers=transformers,
        )        
        grouped_transformer.fit(X, y)
        f_X, f_y = grouped_transformer.transformers[0].fit.call_args_list[-1].args
        assert_frame_equal(f_X, X[[("a1", "f1")]])
        assert array_equal(f_y, y)
        g_X, g_y = grouped_transformer.transformers[1].fit.call_args_list[-1].args
        assert_frame_equal(g_X, X[[("a1", "g1"), ("a2", "g2")]])
        assert array_equal(g_y, y)

    def test_multiindex_columns_subset_transform(self, add_one, subtract_one):
        X = DataFrame(
            columns=MultiIndex.from_arrays(
                [
                    ["a1", "a1", "a1", "a2"],
                    ["f1", "f2", "g1", "g2"],
                ],
                names=["f", "a"],
            ),
            data=zeros(shape=(5, 4), dtype=int),
        )
        y = zeros(5)
        column_ranges = [[("a1", "f1")], [("a1", "g1"), ("a2", "g2")]]
        transformers = [add_one, subtract_one]
        
        grouped_transformer = GroupedTransformer(
            column_ranges=column_ranges,
            transformers=transformers,
        )        
        grouped_transformer.fit(X, y)
        expected = DataFrame(
            columns=Index(
                ["('a1', 'f1')", "('a1', 'g1')", "('a2', 'g2')"],
                name="('f', 'a')",
            ),
            data=[
                [1, -1, -1],
                [1, -1, -1],
                [1, -1, -1],
                [1, -1, -1],
                [1, -1, -1],
            ]
        )
        X_new = grouped_transformer.transform(X)
        assert_frame_equal(X_new, expected)

    def test_multiindex_columns_noflat_subset_fit(self, add_one, subtract_one):
        X = DataFrame(
            columns=MultiIndex.from_arrays(
                [
                    ["a1", "a1", "a1", "a2"],
                    ["f1", "f2", "g1", "g2"],
                ],
                names=["f", "a"],
            ),
            data=zeros(shape=(5, 4), dtype=int),
        )
        y = zeros(5)
        column_ranges = [[("a1", "f1")], [("a1", "g1"), ("a2", "g2")]]
        transformers = [add_one, subtract_one]
        
        grouped_transformer = GroupedTransformer(
            column_ranges=column_ranges,
            transformers=transformers,
            flatten_columns=False,
        )        
        grouped_transformer.fit(X, y)
        f_X, f_y = grouped_transformer.transformers[0].fit.call_args_list[-1].args
        assert_frame_equal(f_X, X[[("a1", "f1")]])
        assert array_equal(f_y, y)
        g_X, g_y = grouped_transformer.transformers[1].fit.call_args_list[-1].args
        assert_frame_equal(g_X, X[[("a1", "g1"), ("a2", "g2")]])
        assert array_equal(g_y, y)

    def test_multiindex_columns_noflat_subset_transform(self, add_one, subtract_one):
        X = DataFrame(
            columns=MultiIndex.from_arrays(
                [
                    ["a1", "a1", "a1", "a2"],
                    ["f1", "f2", "g1", "g2"],
                ],
                names=["f", "a"],
            ),
            data=zeros(shape=(5, 4), dtype=int),
        )
        y = zeros(5)
        column_ranges = [[("a1", "f1")], [("a1", "g1"), ("a2", "g2")]]
        transformers = [add_one, subtract_one]
        
        grouped_transformer = GroupedTransformer(
            column_ranges=column_ranges,
            transformers=transformers,
            flatten_columns=False,
        )        
        grouped_transformer.fit(X, y)
        expected = DataFrame(
            columns=MultiIndex.from_arrays(
                [
                    ["a1", "a1", "a2"],
                    ["f1", "g1", "g2"],
                ],
                names=["f", "a"],
            ),
            data=[
                [1, -1, -1],
                [1, -1, -1],
                [1, -1, -1],
                [1, -1, -1],
                [1, -1, -1],
            ]
        )
        X_new = grouped_transformer.transform(X)
        assert_frame_equal(X_new, expected)

class TestSelectNullScoreOutlier:
    @fixture
    def score_func(self):
        def func(x, y):
            return x[0] + y[0]
        return func

    @fixture
    def null_data(self):
        data = DataFrame(
            index=[
                "sample_1",
                "sample_2",
                "sample_3",
                "sample_4",
            ],
            columns=[
                "null_1", "null_2", "null_3", "null_4", "null_5", "null_y",
            ],
            data=[
                [5, 1, 2, 3, 4,  1],
                [3, 4, 2, 1, 5,  0],
                [2, 4, 3, 1, 5, -1],
                [1, 2, 4, 3, 5,  0],
            ],
        )
        # alpha first_train_row lower_quantile upper_quantile
        #  0.25        sample_1           2.05           5.95
        #  0.25        sample_2           1.05           4.95
        #  0.25        sample_3           0.05           3.95
        #  0.25        sample_4           1.05           4.95
        #  0.8         sample_1           3.48           4.52
        #  0.8         sample_2           2.48           3.52
        #  0.8         sample_3           1.48           2.52
        #  0.8         sample_4           2.48           3.52
        return data

    @fixture
    def train_data(self):
        data = DataFrame(
            index=[
                "sample_4",
                "sample_1",
                "sample_2",
                "sample_3",
            ],
            columns=[
                "train_1", "train_2", "train_3", "train_4", "train_5", "train_6", "train_y",
            ],
            data=[
                [1, 2, 5, 4, 3, 1,   0],
                [3, 4, 5, 1, 2, 3,  -1],
                [2, 5, 3, 1, 4, 5,   1],
                [2, 5, 1, 3, 4, 4,  -1],
            ],
        )
        # alpha train_samples lower_quantile upper_quantile selected_features
        #  0.25           all           2.05           5.95   train_1, train_2, train_6
        #  0.25       1, 2, 3           2.05           5.95   train_1, train_4, train_5, train_6
        #  0.25          2, 3           1.05           4.95   train_2, train_5, train_6
        #  0.25             3           0.05           3.95   train_2, train_3
        #  0.8            all           3.48           4.52   train_1, train_2, train_3, train_5, train_6
        #  0.8        1, 2, 3           3.48           4.52   train_1, train_2, train_4, train_5, train_6
        #  0.8           2, 3           2.48           3.52   train_2, train_3, train_4, train_5, train_6
        #  0.8              3           1.48           2.52   train_1, train_2, train_3, train_5, train_6
        return data

    @fixture
    def zeros_train_data(self):
        data = DataFrame(
            index=[
                "sample_4",
                "sample_1",
                "sample_2",
                "sample_3",
            ],
            columns=[
                "train_1", "train_2", "train_3", "train_4", "train_5", "train_6", "train_y",
            ],
            data=[
                [0, 0, 0, 0, 0, 0, -10],
                [0, 0, 0, 0, 0, 0, -10],
                [0, 0, 0, 0, 0, 0, -10],
                [0, 0, 0, 0, 0, 0, -10],
            ],
        )
        # would always all get selected
        return data

    def test_full_train(self, null_data, train_data, score_func):
        selector = SelectNullScoreOutlier(
            null_data=null_data,
            null_y="null_y",
            score_func=score_func,
            alpha=0.25,
        )
        train_y = train_data.pop("train_y")
        selector.fit(
            X=train_data, y=train_y,
        )
        expected = train_data[["train_1", "train_2", "train_6"]]
        actual = selector.transform(train_data)
        assert_frame_equal(actual, expected)

    def test_subset_train(self, null_data, train_data, score_func):
        selector = SelectNullScoreOutlier(
            null_data=null_data,
            null_y="null_y",
            score_func=score_func,
            alpha=0.25,
        )
        train_data = train_data.iloc[1:] # subsetting train data
        train_y = train_data.pop("train_y")
        selector.fit(
            X=train_data, y=train_y,
        )
        expected = train_data[["train_1", "train_4", "train_5", "train_6"]]
        actual = selector.transform(train_data)
        assert_frame_equal(actual, expected)

    def test_full_predefined_train(self, null_data, train_data, zeros_train_data, score_func):
        selector = SelectNullScoreOutlier(
            null_data=null_data,
            null_y="null_y",
            train_data=train_data,
            train_y="train_y",
            score_func=score_func,
            alpha=0.25,
        )
        zeros_train_y = zeros_train_data.pop("train_y")
        selector.fit(
            X=zeros_train_data, y=zeros_train_y,
        )
        expected = zeros_train_data[["train_1", "train_2", "train_6"]]
        actual = selector.transform(zeros_train_data)
        assert_frame_equal(actual, expected)

    def test_subset_predefined_train(self, null_data, train_data, zeros_train_data, score_func):
        selector = SelectNullScoreOutlier(
            null_data=null_data,
            null_y="null_y",
            train_data=train_data,
            train_y="train_y",
            score_func=score_func,
            alpha=0.25,
        )
        zeros_train_data = zeros_train_data.iloc[2:]
        zeros_train_y = zeros_train_data.pop("train_y")
        selector.fit(
            X=zeros_train_data, y=zeros_train_y,
        )
        expected = zeros_train_data[["train_2", "train_5", "train_6"]]
        actual = selector.transform(zeros_train_data)
        assert_frame_equal(actual, expected)

    def test_alpha_full_train(self, null_data, train_data, score_func):
        selector = SelectNullScoreOutlier(
            null_data=null_data,
            null_y="null_y",
            score_func=score_func,
            alpha=0.8,
        )
        train_y = train_data.pop("train_y")
        selector.fit(
            X=train_data, y=train_y,
        )
        expected = train_data[["train_1", "train_2", "train_3", "train_5", "train_6"]]
        actual = selector.transform(train_data)
        assert_frame_equal(actual, expected)

    def test_alpha_subset_train(self, null_data, train_data, score_func):
        selector = SelectNullScoreOutlier(
            null_data=null_data,
            null_y="null_y",
            score_func=score_func,
            alpha=0.8,
        )
        train_data = train_data.iloc[1:] # subsetting train data
        train_y = train_data.pop("train_y")
        selector.fit(
            X=train_data, y=train_y,
        )
        expected = train_data[["train_1", "train_2", "train_4", "train_5", "train_6"]]
        actual = selector.transform(train_data)
        assert_frame_equal(actual, expected)

class TestSelectPairedNullScoreOutlier:
    @fixture
    def score_func(self):
        def func(x, y):
            return x[0] + y[0]
        return func

    @fixture
    def null_data(self):
        data = DataFrame(
            index=[
                "sample_1",
                "sample_2",
                "sample_3",
                "sample_4",
            ],
            columns=[
                "col_1", "col_2", "col_3", "col_4", "col_5", "null_y",
            ],
            data=[
                [5, 1, 2, 3, 4,  1],
                [3, 4, 2, 1, 5,  0],
                [2, 4, 3, 1, 5, -1],
                [1, 2, 4, 3, 5,  0],
            ],
        )
        # train_samples scores
        #           all   6, 2, 3, 4, 5
        #         1,2,3   6, 2, 3, 4, 5
        #           2,3   3, 4, 2, 1, 5
        #             3   1, 3, 2, 0, 4
        return data

    @fixture
    def train_data(self):
        data = DataFrame(
            index=[
                "sample_4",
                "sample_1",
                "sample_2",
                "sample_3",
            ],
            columns=[
                "col_1", "col_2", "col_3", "col_4", "col_5", "train_y",
            ],
            data=[
                [1, 2, 5, 4, 3,   0],
                [3, 4, 5, 1, 2,  -1],
                [2, 5, 3, 1, 4,   1],
                [2, 5, 1, 3, 4,  -1],
            ],
        )
        # train_samples    scores           delta_scores
        #           all    1, 2, 5, 4, 3    -5,  0,  2,  0, -2
        #       1, 2, 3    2, 3, 4, 0, 1    -4,  1,  1, -4, -4
        #          2, 3    3, 6, 4, 2, 5     0,  2,  2,  1,  0
        #             3    1, 4, 0, 2, 3     0,  1, -2,  2, -1

        # alpha train_samples lower_quantile upper_quantile  selected_columns
        #  0.25           all          -4.85            1.9     col_1, col_3
        #  0.25         1,2,3          -4.              1.      None
        #  0.25           2,3           0.              2.      None
        #  0.25             3          -1.95            1.95    col_3, col_4
        #  0.8            all          -1.04            0.      col_1, col_3, col_5
        #  0.8          1,2,3          -4.             -1.4     col_2, col_3
        #  0.8            2,3           0.48            1.52    col_1, col_2, col_3, col_5
        #  0.8              3          -0.52            0.52    col_2, col_3, col_4, col_5
        return data

    @fixture
    def zeros_train_data(self):
        data = DataFrame(
            index=[
                "sample_4",
                "sample_1",
                "sample_2",
                "sample_3",
            ],
            columns=[
                "col_1", "col_2", "col_3", "col_4", "col_5", "train_y",
            ],
            data=[
                [0, 0, 0, 0, 0, -100],
                [0, 0, 0, 0, 0, -100],
                [0, 0, 0, 0, 0, -100],
                [0, 0, 0, 0, 0, -100],
            ],
        )
        # would always all get selected
        return data

    def test_full_train(self, null_data, train_data, score_func):
        selector = SelectPairedNullScoreOutlier(
            null_data=null_data,
            null_y="null_y",
            score_func=score_func,
            alpha=0.25,
        )
        train_y = train_data.pop("train_y")
        selector.fit(
            X=train_data, y=train_y,
        )
        expected = train_data[["col_1", "col_3"]]
        actual = selector.transform(train_data)
        assert_frame_equal(actual, expected)

    def test_subset_train(self, null_data, train_data, score_func):
        selector = SelectPairedNullScoreOutlier(
            null_data=null_data,
            null_y="null_y",
            score_func=score_func,
            alpha=0.25,
        )
        train_data = train_data.iloc[1:] # subsetting train data
        train_y = train_data.pop("train_y")
        selector.fit(
            X=train_data, y=train_y,
        )
        expected = train_data[[]]
        actual = selector.transform(train_data)
        assert_frame_equal(actual, expected)

    def test_full_predefined_train(self, null_data, train_data, zeros_train_data, score_func):
        selector = SelectPairedNullScoreOutlier(
            null_data=null_data,
            null_y="null_y",
            train_data=train_data,
            train_y="train_y",
            score_func=score_func,
            alpha=0.25,
        )
        zeros_train_y = zeros_train_data.pop("train_y")
        selector.fit(
            X=zeros_train_data, y=zeros_train_y,
        )
        expected = zeros_train_data[["col_1", "col_3"]]
        actual = selector.transform(zeros_train_data)
        assert_frame_equal(actual, expected)

    def test_subset_predefined_train(self, null_data, train_data, zeros_train_data, score_func):
        selector = SelectPairedNullScoreOutlier(
            null_data=null_data,
            null_y="null_y",
            train_data=train_data,
            train_y="train_y",
            score_func=score_func,
            alpha=0.25,
        )
        zeros_train_data = zeros_train_data.iloc[3:]
        zeros_train_y = zeros_train_data.pop("train_y")
        selector.fit(
            X=zeros_train_data, y=zeros_train_y,
        )
        expected = zeros_train_data[["col_3", "col_4"]]
        actual = selector.transform(zeros_train_data)
        assert_frame_equal(actual, expected)

    def test_alpha_full_train(self, null_data, train_data, score_func):
        selector = SelectPairedNullScoreOutlier(
            null_data=null_data,
            null_y="null_y",
            score_func=score_func,
            alpha=0.8,
        )
        train_y = train_data.pop("train_y")
        selector.fit(
            X=train_data, y=train_y,
        )
        expected = train_data[["col_1", "col_3", "col_5"]]
        actual = selector.transform(train_data)
        assert_frame_equal(actual, expected)

    def test_alpha_subset_train(self, null_data, train_data, score_func):
        selector = SelectPairedNullScoreOutlier(
            null_data=null_data,
            null_y="null_y",
            score_func=score_func,
            alpha=0.8,
        )
        train_data = train_data.iloc[1:] # subsetting train data
        train_y = train_data.pop("train_y")
        selector.fit(
            X=train_data, y=train_y,
        )
        expected = train_data[["col_2", "col_3"]]
        actual = selector.transform(train_data)
        assert_frame_equal(actual, expected)




