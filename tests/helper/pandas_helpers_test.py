from numpy import nan
from pandas import DataFrame, Index, MultiIndex
from pandas.testing import assert_frame_equal

from pymmunomics.helper.pandas_helpers import (
    apply_partial_pooled_grouped,
    pivot_pipe_melt,
)


class TestApplyPartialPooledGrouped:
    def test_some_pooled(self):
        data_frame = DataFrame(
            {
                "group1": ["a", "a", "b", "b", "a", "a", "b"],
                "group2": ["a", "a", "a", "a", "b", "b", "a"],
                "val": [1, 2, 3, 5, 7, 11, 13],
            }
        )
        expected = DataFrame(
            {
                "val": [
                    1 * 2 * 7 * 11,
                    3 * 5 * 13,
                    1 * 2 * 3 * 5 * 13,
                    7 * 11,
                ],
            },
            index=MultiIndex.from_arrays(
                [
                    ["a", "b", "pooled", "pooled"],
                    ["pooled", "pooled", "a", "b"],
                ],
                names=["group1", "group2"],
            ),
        )
        actual = apply_partial_pooled_grouped(
            data_frame=data_frame,
            func="prod",
            by=["group1", "group2"],
            pooled=[["group2"], ["group1"]],
        )
        assert_frame_equal(actual, expected)

    def test_all_pooled(self):
        data_frame = DataFrame(
            {
                "group1": ["a", "a", "b", "b", "a", "a", "b"],
                "group2": ["a", "a", "a", "a", "b", "b", "a"],
                "val": [1, 2, 3, 5, 7, 11, 13],
            }
        )
        expected = DataFrame(
            {
                "val": [
                    1 * 2 * 3 * 5 * 7 * 11 * 13,
                ],
            },
            index=MultiIndex.from_arrays(
                [
                    ["pooled"],
                    ["pooled"],
                ],
                names=["group1", "group2"],
            ),
        )
        actual = apply_partial_pooled_grouped(
            data_frame=data_frame,
            func="prod",
            by=["group1", "group2"],
            pooled=[["group1", "group2"]],
        )
        assert_frame_equal(actual, expected)

    def test_none_pooled(self):
        data_frame = DataFrame(
            {
                "group1": ["a", "a", "b", "b", "a", "a", "b"],
                "group2": ["a", "a", "a", "a", "b", "b", "a"],
                "val": [1, 2, 3, 5, 7, 11, 13],
            }
        )
        expected = DataFrame(
            {
                "val": [
                    1 * 2,
                    7 * 11,
                    3 * 5 * 13,
                ],
            },
            index=MultiIndex.from_arrays(
                [
                    ["a", "a", "b"],
                    ["a", "b", "a"],
                ],
                names=["group1", "group2"],
            ),
        )
        actual = apply_partial_pooled_grouped(
            data_frame=data_frame,
            func="prod",
            by=["group1", "group2"],
            pooled=[[]],
        )
        assert_frame_equal(actual, expected)

    def test_mixture_pooled(self):
        data_frame = DataFrame(
            {
                "group1": ["a", "a", "b", "b", "a", "a", "b"],
                "group2": ["a", "a", "a", "a", "b", "b", "a"],
                "val": [1, 2, 3, 5, 7, 11, 13],
            }
        )
        expected = DataFrame(
            {
                "val": [
                    1 * 2,
                    7 * 11,
                    3 * 5 * 13,
                    1 * 2 * 7 * 11,
                    3 * 5 * 13,
                    1 * 2 * 3 * 5 * 7 * 11 * 13,
                ],
            },
            index=MultiIndex.from_arrays(
                [
                    ["a", "a", "b", "a", "b", "pooled"],
                    ["a", "b", "a", "pooled", "pooled", "pooled"],
                ],
                names=["group1", "group2"],
            ),
        )
        actual = apply_partial_pooled_grouped(
            data_frame=data_frame,
            func="prod",
            by=["group1", "group2"],
            pooled=[[], ["group2"], ["group1", "group2"]],
        )
        assert_frame_equal(actual, expected)


class TestPivotPipeMelt:
    def test_multiple_pivot_values(self):
        # pivotted:
        # val1                          val2
        #     col a   b   c                 col a   b   c
        #  idx                           idx
        #   a     1   2   3               a     1   nan 3
        #   b     4   5   nan             b     4   nan nan
        data_frame = DataFrame(
            {
                "col": ["a", "b", "c", "a", "b"],
                "val1": [1, 2, 3, 4, 5],
                "val2": [1, nan, 3, 4, nan],
            },
            index=Index(["a", "a", "a", "b", "b"], name="idx"),
        )
        expected = DataFrame(
            {
                "val1": [1.0, 4.0, 2.0, 5.0, 3.0, 0.0],
                "val2": [1.0, 4.0, 0.0, 0.0, 3.0, 0.0],
            },
            index=Index(
                ["a", "a", "b", "b", "c", "c"],
                name="col",
            ),
        )
        actual = pivot_pipe_melt(
            data_frame=data_frame,
            func=DataFrame.fillna,
            values=["val1", "val2"],
            columns="col",
            value=0,
        )
        assert_frame_equal(actual, expected)

    def test_multiple_pivot_columns(self):
        data_frame = DataFrame(
            {
                "col1": ["a", "a", "b", "c", "c", "b", "b", "c", "c"],
                "col2": ["a", "b", "b", "a", "b", "a", "b", "a", "b"],
                "val": [1, 2, 4, 5, 6, 9, 10, 11, 12],
            },
            index=Index(
                ["a", "a", "a", "a", "a", "b", "b", "b", "b"],
                name="idx",
            ),
        )
        expected = DataFrame(
            {
                "val": [1.0, 0.0, 2.0, 0.0, 0.0, 9.0, 4.0, 10.0, 5.0, 11.0, 6.0, 12.0],
            },
            index=MultiIndex.from_arrays(
                [
                    ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"],
                    ["a", "a", "b", "b", "a", "a", "b", "b", "a", "a", "b", "b"],
                ],
                names=["col1", "col2"],
            ),
        )
        actual = pivot_pipe_melt(
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
            {
                "idx": ["a", "a", "a", "b", "b"],
                "col": ["a", "b", "c", "a", "b"],
                "val": [1, 2, 3, 4, 5],
            },
        )
        expected = DataFrame(
            {
                "val": [1.0, 2.0, 3.0, 4.0, 5.0, 0.0],
            },
            index=MultiIndex.from_arrays(
                [
                    ["a", "a", "a", "b", "b", "b"],
                    ["a", "b", "c", "a", "b", "c"],
                ],
                names=["idx", "col"],
            ),
        )
        actual = pivot_pipe_melt(
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
            {
                "idx1": ["a", "a", "a", "a", "a", "b", "b", "b", "b"],
                "idx2": ["a", "a", "b", "c", "c", "b", "b", "c", "c"],
                "col": ["a", "b", "b", "a", "b", "a", "b", "a", "b"],
                "val": [1, 2, 4, 5, 6, 9, 10, 11, 12],
            },
        )
        expected = DataFrame(
            {
                "val": [1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 9.0, 10.0, 11.0, 12.0],
            },
            index=MultiIndex.from_arrays(
                [
                    ["a", "a", "a", "a", "a", "a", "b", "b", "b", "b"],
                    ["a", "a", "b", "b", "c", "c", "b", "b", "c", "c"],
                    ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
                ],
                names=["idx1", "idx2", "col"],
            ),
        )
        actual = pivot_pipe_melt(
            data_frame=data_frame,
            func=DataFrame.fillna,
            values=["val"],
            columns="col",
            index=["idx1", "idx2"],
            value=0,
        )
        assert_frame_equal(actual, expected)
