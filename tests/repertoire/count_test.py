from pandas import DataFrame, MultiIndex
from pandas.testing import assert_frame_equal

from pymmunomics.repertoire.count import (
    count,
    frequency,
)


class TestCount:
    def test_single_feature_group(self):
        seq = DataFrame(
            columns=["component_1", "component_2", "clonesize"],
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
                names=["feature_group", "feature"],
                tuples=[
                    ("component_1", "a"),
                    ("component_1", "b"),
                    ("component_1", "c"),
                ],
            ),
            columns=["count"],
            data=[
                [11],
                [1100],
                [10000],
            ],
        )
        actual = count(
            seq=seq,
            feature_groups=["component_1"],
            clonesize_column="clonesize",
        )
        assert_frame_equal(actual, expected)
