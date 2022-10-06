from pandas import DataFrame, MultiIndex
from pandas.testing import assert_frame_equal

from pymmunomics.repertoire.count import (
    count,
    frequency,
)


class TestCount:
    def test_single_feature(self):
        repertoire = DataFrame(
            columns=["feature_1", "feature_2", "clonesize"],
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
                names=["feature", "value"],
                tuples=[
                    ("feature_1", "a"),
                    ("feature_1", "b"),
                    ("feature_1", "c"),
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
            repertoire=repertoire,
            features=["feature_1"],
            clonesize="clonesize",
        )
        assert_frame_equal(actual, expected)

    def test_multiple_features(self):
        repertoire = DataFrame(
            columns=["feature_1", "feature_2", "clonesize"],
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
                names=["feature", "value"],
                tuples=[
                    ("feature_1", "a"),
                    ("feature_1", "b"),
                    ("feature_1", "c"),
                    ("feature_2", "x"),
                    ("feature_2", "y"),
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
        actual = count(
            repertoire=repertoire,
            features=["feature_1", "feature_2"],
            clonesize="clonesize",
        )
        assert_frame_equal(actual, expected)

    def test_compound_feature(self):
        repertoire = DataFrame(
            columns=["feature_1", "feature_2", "clonesize"],
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
                names=["feature", "value"],
                tuples=[
                    (("feature_1", "feature_2"), ("a", "x")),
                    (("feature_1", "feature_2"), ("b", "x")),
                    (("feature_1", "feature_2"), ("b", "y")),
                    (("feature_1", "feature_2"), ("c", "y")),
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
        actual = count(
            repertoire=repertoire,
            features=[["feature_1", "feature_2"]],
            clonesize="clonesize",
        )
        assert_frame_equal(actual, expected)

    def test_mixed_features(self):
        repertoire = DataFrame(
            columns=["feature_1", "feature_2", "clonesize"],
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
                names=["feature", "value"],
                tuples=[
                    ("feature_2", "x"),
                    ("feature_2", "y"),
                    (("feature_1", "feature_2"), ("a", "x")),
                    (("feature_1", "feature_2"), ("b", "x")),
                    (("feature_1", "feature_2"), ("b", "y")),
                    (("feature_1", "feature_2"), ("c", "y")),
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
        actual = count(
            repertoire=repertoire,
            features=["feature_2", ["feature_1", "feature_2"]],
            clonesize="clonesize",
        )
        assert_frame_equal(actual, expected)


class TestFrequency:
    def test_single_feature(self):
        repertoire = DataFrame(
            columns=["feature_1", "feature_2", "clonesize"],
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
                names=["feature", "value"],
                tuples=[
                    ("feature_1", "a"),
                    ("feature_1", "b"),
                    ("feature_1", "c"),
                ],
            ),
            columns=["frequency"],
            data=[
                [11 / 11111],
                [1100 / 11111],
                [10000 / 11111],
            ],
        )
        actual = frequency(
            repertoire=repertoire,
            features=["feature_1"],
            clonesize="clonesize",
        )
        assert_frame_equal(actual, expected)

    def test_multiple_features(self):
        repertoire = DataFrame(
            columns=["feature_1", "feature_2", "clonesize"],
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
                names=["feature", "value"],
                tuples=[
                    ("feature_1", "a"),
                    ("feature_1", "b"),
                    ("feature_1", "c"),
                    ("feature_2", "x"),
                    ("feature_2", "y"),
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
        actual = frequency(
            repertoire=repertoire,
            features=["feature_1", "feature_2"],
            clonesize="clonesize",
        )
        assert_frame_equal(actual, expected)

    def test_compound_feature(self):
        repertoire = DataFrame(
            columns=["feature_1", "feature_2", "clonesize"],
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
                names=["feature", "value"],
                tuples=[
                    (("feature_1", "feature_2"), ("a", "x")),
                    (("feature_1", "feature_2"), ("b", "x")),
                    (("feature_1", "feature_2"), ("b", "y")),
                    (("feature_1", "feature_2"), ("c", "y")),
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
        actual = frequency(
            repertoire=repertoire,
            features=[["feature_1", "feature_2"]],
            clonesize="clonesize",
        )
        assert_frame_equal(actual, expected)

    def test_mixed_features(self):
        repertoire = DataFrame(
            columns=["feature_1", "feature_2", "clonesize"],
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
                names=["feature", "value"],
                tuples=[
                    ("feature_2", "x"),
                    ("feature_2", "y"),
                    (("feature_1", "feature_2"), ("a", "x")),
                    (("feature_1", "feature_2"), ("b", "x")),
                    (("feature_1", "feature_2"), ("b", "y")),
                    (("feature_1", "feature_2"), ("c", "y")),
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
        actual = frequency(
            repertoire=repertoire,
            features=["feature_2", ["feature_1", "feature_2"]],
            clonesize="clonesize",
        )
        assert_frame_equal(actual, expected)
