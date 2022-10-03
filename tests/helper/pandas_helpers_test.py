from pandas import DataFrame
from pandas import MultiIndex
from pandas.testing import assert_frame_equal

from pymmunomics.helper.pandas_helpers import apply_partial_pooled_grouped


class TestApplyPartialPooledGrouped:
    def test_expected(self):
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
