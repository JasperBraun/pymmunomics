from unittest.mock import MagicMock

from numpy import array_equal, zeros
from pandas import DataFrame, Index, MultiIndex
from pandas.testing import assert_frame_equal
from pytest import fixture

from pymmunomics.experimental.ml.feature_selection import (
    GroupedTransformer,
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
