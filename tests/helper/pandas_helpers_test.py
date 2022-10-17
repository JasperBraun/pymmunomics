from numpy import nan
from pandas import DataFrame, Index, MultiIndex
from pandas.testing import assert_frame_equal
from pytest import raises

from pymmunomics.helper.pandas_helpers import (
    assert_groups_equal,
    concat_partial_groupby_apply,
    concat_pivot_pipe_melt,
    concat_weighted_value_counts,
)


class TestAssertGroupsEqual:
    def test_nopipe(self):
        data_frame = DataFrame(
            columns=["g1", "g2", "val1", "val2"],
            data=[
                ["a", "a", 1, 1],
                ["a", "a", 2, 2],
                ["a", "b", 1, -1],
                ["a", "b", 2, 2],
                ["b", "a", 1, 1],
                ["b", "a", 2, 2],
            ],
        )
        with raises(AssertionError):
            assert_groups_equal(
                data_frame=data_frame,
                groupby_kwargs={"by": ["g1", "g2"]},
            )

    def test_equal(self):
        data_frame = DataFrame(
            columns=["g1", "g2", "val1", "val2"],
            data=[
                ["a", "a", 1, 1],
                ["a", "a", 2, 2],
                ["a", "b", 1, -1],
                ["a", "b", 2, 2],
                ["b", "a", 1, 1],
                ["b", "a", 2, 2],
            ],
        )
        assert_groups_equal(
            data_frame=data_frame,
            groupby_kwargs={"by": ["g1", "g2"]},
            group_pipe=lambda df: df[["val1"]].reset_index(drop=True),
        )

    def test_different(self):
        data_frame = DataFrame(
            columns=["g1", "g2", "val1", "val2"],
            data=[
                ["a", "a", 1, 1],
                ["a", "a", 2, 2],
                ["a", "b", 1, -1],
                ["a", "b", 2, 2],
                ["b", "a", 1, 1],
                ["b", "a", 2, 2],
            ],
        )
        with raises(AssertionError):
            assert_groups_equal(
                data_frame=data_frame,
                groupby_kwargs={"by": ["g1", "g2"]},
                group_pipe=lambda df: df[["val2"]].reset_index(drop=True),
            )

    def test_float_close_assert_frame_equal_kwargs(self):
        data_frame = DataFrame(
            columns=["g1", "g2", "val1", "val2"],
            data=[
                ["a", "a", 1, 1],
                ["a", "a", 2, 2],
                ["a", "b", 1 + 1e-3, -1],
                ["a", "b", 2, 2],
                ["b", "a", 1, 1],
                ["b", "a", 2, 2],
            ],
        )
        assert_groups_equal(
            data_frame=data_frame,
            groupby_kwargs={"by": ["g1", "g2"]},
            group_pipe=lambda df: df[["val1"]].reset_index(drop=True),
            assert_frame_equal_kwargs={"check_exact": False, "rtol": 0, "atol": 1e-2},
        )

    def test_float_differ_assert_frame_equal_kwargs(self):
        data_frame = DataFrame(
            columns=["g1", "g2", "val1", "val2"],
            data=[
                ["a", "a", 1, 1],
                ["a", "a", 2, 2],
                ["a", "b", 1 + 1e-3, -1],
                ["a", "b", 2, 2],
                ["b", "a", 1, 1],
                ["b", "a", 2, 2],
            ],
        )
        with raises(AssertionError):
            assert_groups_equal(
                data_frame=data_frame,
                groupby_kwargs={"by": ["g1", "g2"]},
                group_pipe=lambda df: (df[["val1"]].reset_index(drop=True)),
                assert_frame_equal_kwargs={"check_exact": False, "rtol": 0, "atol": 1e-4},
            )


class TestConcatPartialGroupbyApply:
    def test_some_pooled(self):
        data_frame = DataFrame(
            columns=["group1", "group2", "val"],
            data=[
                ["a", "a", 1],
                ["a", "a", 2],
                ["b", "a", 3],
                ["b", "a", 5],
                ["a", "b", 7],
                ["a", "b", 11],
                ["b", "a", 13],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["group1", "group2"],
                tuples=[
                    ["a", "pooled"],
                    ["b", "pooled"],
                    ["pooled", "a"],
                    ["pooled", "b"],
                ],
            ),
            columns=["val"],
            data=[
                [1 * 2 * 7 * 11],
                [3 * 5 * 13],
                [1 * 2 * 3 * 5 * 13],
                [7 * 11],
            ],
        )
        actual = concat_partial_groupby_apply(
            data_frame=data_frame,
            func=lambda df: df[["val"]].prod(),
            by=["group1", "group2"],
            pooled=[["group2"], ["group1"]],
        )
        assert_frame_equal(actual, expected)

    def test_all_pooled(self):
        data_frame = DataFrame(
            columns=["group1", "group2", "val"],
            data=[
                ["a", "a", 1],
                ["a", "a", 2],
                ["b", "a", 3],
                ["b", "a", 5],
                ["a", "b", 7],
                ["a", "b", 11],
                ["b", "a", 13],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["group1", "group2"],
                tuples=[
                    ("pooled", "pooled"),
                ],
            ),
            columns=["val"],
            data=[
                [1 * 2 * 3 * 5 * 7 * 11 * 13],
            ],
        )
        actual = concat_partial_groupby_apply(
            data_frame=data_frame,
            func=lambda df: df[["val"]].prod(),
            by=["group1", "group2"],
            pooled=[["group1", "group2"]],
        )
        assert_frame_equal(actual, expected)

    def test_none_pooled(self):
        data_frame = DataFrame(
            columns=["group1", "group2", "val"],
            data=[
                ["a", "a", 1],
                ["a", "a", 2],
                ["b", "a", 3],
                ["b", "a", 5],
                ["a", "b", 7],
                ["a", "b", 11],
                ["b", "a", 13],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["group1", "group2"],
                tuples=[
                    ("a", "a"),
                    ("a", "b"),
                    ("b", "a"),
                ],
            ),
            columns=["val"],
            data=[
                [1 * 2],
                [7 * 11],
                [3 * 5 * 13],
            ],
        )
        actual = concat_partial_groupby_apply(
            data_frame=data_frame,
            func=lambda df: df[["val"]].prod(),
            by=["group1", "group2"],
            pooled=[[]],
        )
        assert_frame_equal(actual, expected)

    def test_mixture_pooled(self):
        data_frame = DataFrame(
            columns=["group1", "group2", "val"],
            data=[
                ["a", "a", 1],
                ["a", "a", 2],
                ["b", "a", 3],
                ["b", "a", 5],
                ["a", "b", 7],
                ["a", "b", 11],
                ["b", "a", 13],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["group1", "group2"],
                tuples=[
                    ("a", "a"),
                    ("a", "b"),
                    ("b", "a"),
                    ("a", "pooled"),
                    ("b", "pooled"),
                    ("pooled", "pooled"),
                ],
            ),
            columns=["val"],
            data=[
                [1 * 2],
                [7 * 11],
                [3 * 5 * 13],
                [1 * 2 * 7 * 11],
                [3 * 5 * 13],
                [1 * 2 * 3 * 5 * 7 * 11 * 13],
            ],
        )
        actual = concat_partial_groupby_apply(
            data_frame=data_frame,
            func=lambda df: df[["val"]].prod(),
            by=["group1", "group2"],
            pooled=[[], ["group2"], ["group1", "group2"]],
        )
        assert_frame_equal(actual, expected)


class TestConcatPivotPipeMelt:
    def test_multiple_pivot_values(self):
        # pivotted:
        # val1                          val2
        #     col a   b   c                 col a   b   c
        #  idx                           idx
        #   a     1   2   3               a     1   nan 3
        #   b     4   5   nan             b     4   nan nan
        data_frame = DataFrame(
            index=Index(
                name="idx",
                data=[
                    "a",
                    "a",
                    "a",
                    "b",
                    "b",
                ],
            ),
            columns=["col", "val1", "val2"],
            data=[
                ["a", 1, 1],
                ["b", 2, nan],
                ["c", 3, 3],
                ["a", 4, 4],
                ["b", 5, nan],
            ],
        )
        expected = DataFrame(
            index=Index(
                name="col",
                data=[
                    "a",
                    "a",
                    "b",
                    "b",
                    "c",
                    "c",
                ],
            ),
            columns=["val1", "val2"],
            data=[
                [1., 1.],
                [4., 4.],
                [2., 0.],
                [5., 0.],
                [3., 3.],
                [0., 0.],
            ],
        )
        actual = concat_pivot_pipe_melt(
            data_frame=data_frame,
            func=DataFrame.fillna,
            values=["val1", "val2"],
            columns="col",
            value=0,
        )
        assert_frame_equal(actual, expected)

    def test_multiple_pivot_columns(self):
        data_frame = DataFrame(
            index=Index(
                name="idx",
                data=[
                    "a",
                    "a",
                    "a",
                    "a",
                    "a",
                    "b",
                    "b",
                    "b",
                    "b",
                ],
            ),
            columns=["col1", "col2", "val"],
            data=[
                ["a", "a", 1],
                ["a", "b", 2],
                ["b", "b", 4],
                ["c", "a", 5],
                ["c", "b", 6],
                ["b", "a", 9],
                ["b", "b", 10],
                ["c", "a", 11],
                ["c", "b", 12],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["col1", "col2"],
                tuples=[
                    ("a", "a"),
                    ("a", "a"),
                    ("a", "b"),
                    ("a", "b"),
                    ("b", "a"),
                    ("b", "a"),
                    ("b", "b"),
                    ("b", "b"),
                    ("c", "a"),
                    ("c", "a"),
                    ("c", "b"),
                    ("c", "b"),
                ],
            ),
            columns=["val"],
            data=[
                [1.0],
                [0.0],
                [2.0],
                [0.0],
                [0.0],
                [9.0],
                [4.0],
                [10.0],
                [5.0],
                [11.0],
                [6.0],
                [12.0],
            ],
        )
        actual = concat_pivot_pipe_melt(
            data_frame=data_frame,
            func=DataFrame.fillna,
            values=["val"],
            columns=["col1", "col2"],
            value=0,
        )
        assert_frame_equal(actual, expected)

    def test_single_pivot_index(self):
        # pivotted:
        # val
        #     col a   b   c
        #  idx
        #   a     1   2   3
        #   b     4   5   nan
        data_frame = DataFrame(
            columns=["idx", "col", "val"],
            data=[
                ["a", "a", 1],
                ["a", "b", 2],
                ["a", "c", 3],
                ["b", "a", 4],
                ["b", "b", 5],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["idx", "col"],
                tuples=[
                    ("a", "a"),
                    ("a", "b"),
                    ("a", "c"),
                    ("b", "a"),
                    ("b", "b"),
                    ("b", "c"),
                ],
            ),
            columns=["val"],
            data=[
                [1.0],
                [2.0],
                [3.0],
                [4.0],
                [5.0],
                [0.0],
            ],
        )
        actual = concat_pivot_pipe_melt(
            data_frame=data_frame,
            func=DataFrame.fillna,
            values=["val"],
            columns="col",
            index="idx",
            value=0,
        )
        assert_frame_equal(actual, expected)

    def test_multiple_pivot_index(self):
        # pivotted:
        #     col      a   b
        #  idx1 idx2
        #   a    a     1   2
        #   a    b     nan 4
        #   a    c     5   6
        #   b    b     9   10
        #   b    c     11  12
        data_frame = DataFrame(
            columns=["idx1", "idx2", "col", "val"],
            data=[
                ["a", "a", "a", 1],
                ["a", "a", "b", 2],
                ["a", "b", "b", 4],
                ["a", "c", "a", 5],
                ["a", "c", "b", 6],
                ["b", "b", "a", 9],
                ["b", "b", "b", 10],
                ["b", "c", "a", 11],
                ["b", "c", "b", 12],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["idx1", "idx2", "col"],
                tuples=[
                    ('a', 'a', 'a'),
                    ('a', 'a', 'b'),
                    ('a', 'b', 'a'),
                    ('a', 'b', 'b'),
                    ('a', 'c', 'a'),
                    ('a', 'c', 'b'),
                    ('b', 'b', 'a'),
                    ('b', 'b', 'b'),
                    ('b', 'c', 'a'),
                    ('b', 'c', 'b'),
                ],
            ),
            columns=["val"],
            data=[
                [ 1.0],
                [ 2.0],
                [ 0.0],
                [ 4.0],
                [ 5.0],
                [ 6.0],
                [ 9.0],
                [10.0],
                [11.0],
                [12.0],
            ],
        )
        actual = concat_pivot_pipe_melt(
            data_frame=data_frame,
            func=DataFrame.fillna,
            values=["val"],
            columns="col",
            index=["idx1", "idx2"],
            value=0,
        )
        assert_frame_equal(actual, expected)

    def test_mixture(self):
        data_frame = DataFrame(
            columns=["idx", "col", "val1", "val2"],
            data=[
                ["a", "a", 1, 1],
                ["a", "b", 2, nan],
                ["a", "c", 3, 3],
                ["b", "a", 4, 4],
                ["b", "b", 5, nan],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["idx", "col"],
                tuples=[
                    ("a", "a"),
                    ("a", "b"),
                    ("a", "c"),
                    ("b", "a"),
                    ("b", "b"),
                    ("b", "c"),
                ],
            ),
            columns=["val1", "val2"],
            data=[
                [1.0, 1.0],
                [2.0, 0.0],
                [3.0, 3.0],
                [4.0, 4.0],
                [5.0, 0.0],
                [0.0, 0.0],
            ],
        )
        actual = concat_pivot_pipe_melt(
            data_frame=data_frame,
            func=DataFrame.fillna,
            values=["val1", "val2"],
            columns="col",
            index=["idx"],
            value=0,
        )
        assert_frame_equal(actual, expected)

class TestConcatWeightedValueCounts:
    def test_single_column(self):
        data_frame = DataFrame(
            columns=["column_1", "column_2", "weight"],
            data=[
                ["a", "x", 1],
                ["a", "x", 10],
                ["b", "x", 100],
                ["b", "y", 1000],
                ["c", "y", 10000],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["columns", "values"],
                tuples=[
                    ("column_1", "a"),
                    ("column_1", "b"),
                    ("column_1", "c"),
                ],
            ),
            columns=["count"],
            data=[
                [11],
                [1100],
                [10000],
            ],
        )
        actual = concat_weighted_value_counts(
            data_frame=data_frame,
            subsets=["column_1"],
            weight="weight",
        )
        assert_frame_equal(actual, expected)

    def test_multiple_columns(self):
        data_frame = DataFrame(
            columns=["column_1", "column_2", "weight"],
            data=[
                ["a", "x", 1],
                ["a", "x", 10],
                ["b", "x", 100],
                ["b", "y", 1000],
                ["c", "y", 10000],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["columns", "values"],
                tuples=[
                    ("column_1", "a"),
                    ("column_1", "b"),
                    ("column_1", "c"),
                    ("column_2", "x"),
                    ("column_2", "y"),
                ],
            ),
            columns=["count"],
            data=[
                [11],
                [1100],
                [10000],
                [111],
                [11000],
            ],
        )
        actual = concat_weighted_value_counts(
            data_frame=data_frame,
            subsets=["column_1", "column_2"],
            weight="weight",
        )
        assert_frame_equal(actual, expected)

    def test_compound_column(self):
        data_frame = DataFrame(
            columns=["column_1", "column_2", "weight"],
            data=[
                ["a", "x", 1],
                ["a", "x", 10],
                ["b", "x", 100],
                ["b", "y", 1000],
                ["c", "y", 10000],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["columns", "values"],
                tuples=[
                    (("column_1", "column_2"), ("a", "x")),
                    (("column_1", "column_2"), ("b", "x")),
                    (("column_1", "column_2"), ("b", "y")),
                    (("column_1", "column_2"), ("c", "y")),
                ],
            ),
            columns=["count"],
            data=[
                [11],
                [100],
                [1000],
                [10000],
            ],
        )
        actual = concat_weighted_value_counts(
            data_frame=data_frame,
            subsets=[["column_1", "column_2"]],
            weight="weight",
        )
        assert_frame_equal(actual, expected)

    def test_mixed_columns(self):
        data_frame = DataFrame(
            columns=["column_1", "column_2", "weight"],
            data=[
                ["a", "x", 1],
                ["a", "x", 10],
                ["b", "x", 100],
                ["b", "y", 1000],
                ["c", "y", 10000],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["columns", "values"],
                tuples=[
                    ("column_2", "x"),
                    ("column_2", "y"),
                    (("column_1", "column_2"), ("a", "x")),
                    (("column_1", "column_2"), ("b", "x")),
                    (("column_1", "column_2"), ("b", "y")),
                    (("column_1", "column_2"), ("c", "y")),
                ],
            ),
            columns=["count"],
            data=[
                [111],
                [11000],
                [11],
                [100],
                [1000],
                [10000],
            ],
        )
        actual = concat_weighted_value_counts(
            data_frame=data_frame,
            subsets=["column_2", ["column_1", "column_2"]],
            weight="weight",
        )
        assert_frame_equal(actual, expected)

    def test_single_column_normalize(self):
        data_frame = DataFrame(
            columns=["column_1", "column_2", "weight"],
            data=[
                ["a", "x", 1],
                ["a", "x", 10],
                ["b", "x", 100],
                ["b", "y", 1000],
                ["c", "y", 10000],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["columns", "values"],
                tuples=[
                    ("column_1", "a"),
                    ("column_1", "b"),
                    ("column_1", "c"),
                ],
            ),
            columns=["frequency"],
            data=[
                [11 / 11111],
                [1100 / 11111],
                [10000 / 11111],
            ],
        )
        actual = concat_weighted_value_counts(
            data_frame=data_frame,
            subsets=["column_1"],
            weight="weight",
            normalize=True,
        )
        assert_frame_equal(actual, expected)

    def test_multiple_columns_normalize(self):
        data_frame = DataFrame(
            columns=["column_1", "column_2", "weight"],
            data=[
                ["a", "x", 1],
                ["a", "x", 10],
                ["b", "x", 100],
                ["b", "y", 1000],
                ["c", "y", 10000],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["columns", "values"],
                tuples=[
                    ("column_1", "a"),
                    ("column_1", "b"),
                    ("column_1", "c"),
                    ("column_2", "x"),
                    ("column_2", "y"),
                ],
            ),
            columns=["frequency"],
            data=[
                [11 / 11111],
                [1100 / 11111],
                [10000 / 11111],
                [111 / 11111],
                [11000 / 11111],
            ],
        )
        actual = concat_weighted_value_counts(
            data_frame=data_frame,
            subsets=["column_1", "column_2"],
            weight="weight",
            normalize=True,
        )
        assert_frame_equal(actual, expected)

    def test_compound_column_normalize(self):
        data_frame = DataFrame(
            columns=["column_1", "column_2", "weight"],
            data=[
                ["a", "x", 1],
                ["a", "x", 10],
                ["b", "x", 100],
                ["b", "y", 1000],
                ["c", "y", 10000],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["columns", "values"],
                tuples=[
                    (("column_1", "column_2"), ("a", "x")),
                    (("column_1", "column_2"), ("b", "x")),
                    (("column_1", "column_2"), ("b", "y")),
                    (("column_1", "column_2"), ("c", "y")),
                ],
            ),
            columns=["frequency"],
            data=[
                [11 / 11111],
                [100 / 11111],
                [1000 / 11111],
                [10000 / 11111],
            ],
        )
        actual = concat_weighted_value_counts(
            data_frame=data_frame,
            subsets=[["column_1", "column_2"]],
            weight="weight",
            normalize=True,
        )
        assert_frame_equal(actual, expected)

    def test_mixed_columns_normalize(self):
        data_frame = DataFrame(
            columns=["column_1", "column_2", "weight"],
            data=[
                ["a", "x", 1],
                ["a", "x", 10],
                ["b", "x", 100],
                ["b", "y", 1000],
                ["c", "y", 10000],
            ],
        )
        expected = DataFrame(
            index=MultiIndex.from_tuples(
                names=["columns", "values"],
                tuples=[
                    ("column_2", "x"),
                    ("column_2", "y"),
                    (("column_1", "column_2"), ("a", "x")),
                    (("column_1", "column_2"), ("b", "x")),
                    (("column_1", "column_2"), ("b", "y")),
                    (("column_1", "column_2"), ("c", "y")),
                ],
            ),
            columns=["frequency"],
            data=[
                [111 / 11111],
                [11000 / 11111],
                [11 / 11111],
                [100 / 11111],
                [1000 / 11111],
                [10000 / 11111],
            ],
        )
        actual = concat_weighted_value_counts(
            data_frame=data_frame,
            subsets=["column_2", ["column_1", "column_2"]],
            weight="weight",
            normalize=True,
        )
        assert_frame_equal(actual, expected)
