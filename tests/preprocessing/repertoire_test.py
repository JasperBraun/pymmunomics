from pandas import DataFrame
from pandas.testing import assert_frame_equal

from pymmunomics.preprocessing.repertoire import count_clonotype_features

class TestCountClonotypes:
    def test_no_pools_no_shared_feature_groups(self):
        repertoire = DataFrame(
            columns=[
                "g1", "g2", "f1", "f2", "clonesize",
            ],
            data=[
                ["a", "a", 1, "x", 1],
                ["a", "b", 2, "x", 2],
                ["a", "a", 3, "y", 10],
                ["b", "a", 1, "x", 20],
                ["a", "b", 2, "y", 100],
                ["a", "b", 3, "x", 200],
                ["a", "a", 1, "y", 1000],
            ],
        )
        expected = DataFrame(
            columns=[
                "g1", "g2", "feature", "feature_value", "frequency",
            ],
            data=[
                ["a", "a", "f1", "1", 1001 / 1011],
                ["a", "a", "f1", "3",   10 / 1011],
                ["a", "a", "f2", "x",    1 / 1011],
                ["a", "a", "f2", "y", 1010 / 1011],
                ["a", "b", "f1", "2",  102 /  302],
                ["a", "b", "f1", "3",  200 /  302],
                ["a", "b", "f2", "x",  202 /  302],
                ["a", "b", "f2", "y",  100 /  302],
                ["b", "a", "f1", "1",   20 /   20],
                ["b", "a", "f2", "x",   20 /   20],
            ],
        )
        actual = count_clonotype_features(
            repertoire=repertoire,
            repertoire_groups=["g1", "g2"],
            clonotype_features=["f1", "f2"],
            clonesize="clonesize",
        )
        assert_frame_equal(actual, expected)

    def test_partial_pools_no_shared_feature_groups(self):
        repertoire = DataFrame(
            columns=[
                "g1", "g2", "f1", "f2", "clonesize",
            ],
            data=[
                ["a", "a", 1, "x", 1],
                ["a", "b", 2, "x", 2],
                ["a", "a", 3, "y", 10],
                ["b", "a", 1, "x", 20],
                ["a", "b", 2, "y", 100],
                ["a", "b", 3, "x", 200],
                ["a", "a", 1, "y", 1000],
            ],
        )
        expected = DataFrame(
            columns=[
                "g1", "g2", "feature", "feature_value", "frequency",
            ],
            data=[
                ["a", "a", "f1", "1", 1001 / 1011],
                ["a", "a", "f1", "3",   10 / 1011],
                ["a", "a", "f2", "x",    1 / 1011],
                ["a", "a", "f2", "y", 1010 / 1011],
                ["a", "b", "f1", "2",  102 /  302],
                ["a", "b", "f1", "3",  200 /  302],
                ["a", "b", "f2", "x",  202 /  302],
                ["a", "b", "f2", "y",  100 /  302],
                ["b", "a", "f1", "1",   20 /   20],
                ["b", "a", "f2", "x",   20 /   20],
                ["pooled", "a", "f1", "1", 1021 / 1031],
                ["pooled", "a", "f1", "3",   10 / 1031],
                ["pooled", "a", "f2", "x",   21 / 1031],
                ["pooled", "a", "f2", "y", 1010 / 1031],
                ["pooled", "b", "f1", "2",  102 /  302],
                ["pooled", "b", "f1", "3",  200 /  302],
                ["pooled", "b", "f2", "x",  202 /  302],
                ["pooled", "b", "f2", "y",  100 /  302],
                ["pooled", "pooled", "f1", "1", 1021 / 1333],
                ["pooled", "pooled", "f1", "2",  102 / 1333],
                ["pooled", "pooled", "f1", "3",  210 / 1333],
                ["pooled", "pooled", "f2", "x",  223 / 1333],
                ["pooled", "pooled", "f2", "y", 1110 / 1333],
            ],
        )
        actual = count_clonotype_features(
            repertoire=repertoire,
            repertoire_groups=["g1", "g2"],
            clonotype_features=["f1", "f2"],
            clonesize="clonesize",
            partial_repertoire_pools=[[], ["g1"], ["g1", "g2"]],
        )
        assert_frame_equal(actual, expected)

    def test_no_pools_shared_feature_group(self):
        repertoire = DataFrame(
            columns=[
                "g1", "g2", "f1", "f2", "clonesize",
            ],
            data=[
                ["a", "a", 1, "x", 1],
                ["a", "b", 2, "x", 2],
                ["a", "a", 3, "y", 10],
                ["b", "a", 1, "x", 20],
                ["a", "b", 2, "y", 100],
                ["a", "b", 3, "x", 200],
                ["a", "a", 1, "y", 1000],
            ],
        )
        expected = DataFrame(
            columns=[
                "g1", "g2", "feature", "feature_value", "frequency",
            ],
            data=[
                ["a", "a", "f1", "1", 1001 / 1011],
                ["a", "a", "f1", "2",    0 / 1011],
                ["a", "a", "f1", "3",   10 / 1011],
                ["a", "a", "f2", "x",    1 / 1011],
                ["a", "a", "f2", "y", 1010 / 1011],
                ["a", "b", "f1", "1",    0 /  302],
                ["a", "b", "f1", "2",  102 /  302],
                ["a", "b", "f1", "3",  200 /  302],
                ["a", "b", "f2", "x",  202 /  302],
                ["a", "b", "f2", "y",  100 /  302],
                ["b", "a", "f1", "1",   20 /   20],
                ["b", "a", "f2", "x",   20 /   20],
            ],
        )
        actual = count_clonotype_features(
            repertoire=repertoire,
            repertoire_groups=["g1", "g2"],
            clonotype_features=["f1", "f2"],
            clonesize="clonesize",
            shared_clonotype_feature_groups=["g2"],
        )
        assert_frame_equal(actual, expected)

    def test_partial_pools_shared_feature_groups(self):
        repertoire = DataFrame(
            columns=[
                "g1", "g2", "f1", "f2", "clonesize",
            ],
            data=[
                ["a", "a", 1, "x", 1],
                ["a", "b", 2, "x", 2],
                ["a", "a", 3, "y", 10],
                ["b", "a", 1, "x", 20],
                ["a", "b", 2, "y", 100],
                ["a", "b", 3, "x", 200],
                ["a", "a", 1, "y", 1000],
            ],
        )
        expected = DataFrame(
            columns=[
                "g1", "g2", "feature", "feature_value", "frequency",
            ],
            data=[
                ["a", "a", "f1", "1", 1001 / 1011],
                ["a", "a", "f1", "3",   10 / 1011],
                ["a", "a", "f2", "x",    1 / 1011],
                ["a", "a", "f2", "y", 1010 / 1011],
                ["a", "b", "f1", "2",  102 /  302],
                ["a", "b", "f1", "3",  200 /  302],
                ["a", "b", "f2", "x",  202 /  302],
                ["a", "b", "f2", "y",  100 /  302],
                ["b", "a", "f1", "1",   20 /   20],
                ["b", "a", "f1", "3",    0 /   20],
                ["b", "a", "f2", "x",   20 /   20],
                ["b", "a", "f2", "y",    0 /   20],
                ["pooled", "a", "f1", "1", 1021 / 1031],
                ["pooled", "a", "f1", "3",   10 / 1031],
                ["pooled", "a", "f2", "x",   21 / 1031],
                ["pooled", "a", "f2", "y", 1010 / 1031],
                ["pooled", "b", "f1", "2",  102 /  302],
                ["pooled", "b", "f1", "3",  200 /  302],
                ["pooled", "b", "f2", "x",  202 /  302],
                ["pooled", "b", "f2", "y",  100 /  302],
                ["pooled", "pooled", "f1", "1", 1021 / 1333],
                ["pooled", "pooled", "f1", "2",  102 / 1333],
                ["pooled", "pooled", "f1", "3",  210 / 1333],
                ["pooled", "pooled", "f2", "x",  223 / 1333],
                ["pooled", "pooled", "f2", "y", 1110 / 1333],
            ],
        )
        actual = count_clonotype_features(
            repertoire=repertoire,
            repertoire_groups=["g1", "g2"],
            clonotype_features=["f1", "f2"],
            clonesize="clonesize",
            partial_repertoire_pools=[[], ["g1"], ["g1", "g2"]],
            shared_clonotype_feature_groups=["g1"],
        )
        assert_frame_equal(actual, expected)

    def test_absolute_counts(self):
        repertoire = DataFrame(
            columns=[
                "g1", "g2", "f1", "f2", "clonesize",
            ],
            data=[
                ["a", "a", 1, "x", 1],
                ["a", "b", 2, "x", 2],
                ["a", "a", 3, "y", 10],
                ["b", "a", 1, "x", 20],
                ["a", "b", 2, "y", 100],
                ["a", "b", 3, "x", 200],
                ["a", "a", 1, "y", 1000],
            ],
        )
        expected = DataFrame(
            columns=[
                "g1", "g2", "feature", "feature_value", "count",
            ],
            data=[
                ["a", "a", "f1", "1", 1001],
                ["a", "a", "f1", "3",   10],
                ["a", "a", "f2", "x",    1],
                ["a", "a", "f2", "y", 1010],
                ["a", "b", "f1", "2",  102],
                ["a", "b", "f1", "3",  200],
                ["a", "b", "f2", "x",  202],
                ["a", "b", "f2", "y",  100],
                ["b", "a", "f1", "1",   20],
                ["b", "a", "f2", "x",   20],
            ],
        )
        actual = count_clonotype_features(
            repertoire=repertoire,
            repertoire_groups=["g1", "g2"],
            clonotype_features=["f1", "f2"],
            clonesize="clonesize",
            normalize=False
        )
        assert_frame_equal(actual, expected)

    def test_absolute_counts_partial_pools_shared_feature_groups(self):
        repertoire = DataFrame(
            columns=[
                "g1", "g2", "f1", "f2", "clonesize",
            ],
            data=[
                ["a", "a", 1, "x", 1],
                ["a", "b", 2, "x", 2],
                ["a", "a", 3, "y", 10],
                ["b", "a", 1, "x", 20],
                ["a", "b", 2, "y", 100],
                ["a", "b", 3, "x", 200],
                ["a", "a", 1, "y", 1000],
            ],
        )
        expected = DataFrame(
            columns=[
                "g1", "g2", "feature", "feature_value", "count",
            ],
            data=[
                ["a", "a", "f1", "1", 1001],
                ["a", "a", "f1", "3",   10],
                ["a", "a", "f2", "x",    1],
                ["a", "a", "f2", "y", 1010],
                ["a", "b", "f1", "2",  102],
                ["a", "b", "f1", "3",  200],
                ["a", "b", "f2", "x",  202],
                ["a", "b", "f2", "y",  100],
                ["b", "a", "f1", "1",   20],
                ["b", "a", "f1", "3",    0],
                ["b", "a", "f2", "x",   20],
                ["b", "a", "f2", "y",    0],
                ["pooled", "a", "f1", "1", 1021],
                ["pooled", "a", "f1", "3",   10],
                ["pooled", "a", "f2", "x",   21],
                ["pooled", "a", "f2", "y", 1010],
                ["pooled", "b", "f1", "2",  102],
                ["pooled", "b", "f1", "3",  200],
                ["pooled", "b", "f2", "x",  202],
                ["pooled", "b", "f2", "y",  100],
                ["pooled", "pooled", "f1", "1", 1021],
                ["pooled", "pooled", "f1", "2",  102],
                ["pooled", "pooled", "f1", "3",  210],
                ["pooled", "pooled", "f2", "x",  223],
                ["pooled", "pooled", "f2", "y", 1110],
            ],
        )
        actual = count_clonotype_features(
            repertoire=repertoire,
            repertoire_groups=["g1", "g2"],
            clonotype_features=["f1", "f2"],
            clonesize="clonesize",
            partial_repertoire_pools=[[], ["g1"], ["g1", "g2"]],
            normalize=False,
            shared_clonotype_feature_groups=["g1"],
        )
        assert_frame_equal(actual, expected)

    def test_partial_pools_shared_feature_groups_docs_example(self):
        repertoire = DataFrame(
            columns=[
                "g1", "g2", "f1", "f2", "clonesize",
            ],
            data=[
                ["a", "a", 1, "x", 1],
                ["a", "b", 2, "x", 2],
                ["a", "a", 3, "y", 10],
                ["b", "a", 1, "x", 20],
                ["a", "b", 2, "y", 100],
                ["a", "b", 3, "x", 200],
                ["a", "a", 1, "y", 1000],
            ],
        )
        expected = DataFrame(
            columns=[
                "g1", "g2", "feature", "feature_value", "frequency",
            ],
            data=[
                ["a", "a", "f1", "1", 1001 / 1011],
                ["a", "a", "f1", "3",   10 / 1011],
                ["a", "a", "f2", "x",    1 / 1011],
                ["a", "a", "f2", "y", 1010 / 1011],
                ["a", "b", "f1", "2",  102 /  302],
                ["a", "b", "f1", "3",  200 /  302],
                ["a", "b", "f2", "x",  202 /  302],
                ["a", "b", "f2", "y",  100 /  302],
                ["b", "a", "f1", "1",   20 /   20],
                ["b", "a", "f1", "3",    0 /   20],
                ["b", "a", "f2", "x",   20 /   20],
                ["b", "a", "f2", "y",    0 /   20],
                ["pooled", "a", "f1", "1", 1021 / 1031],
                ["pooled", "a", "f1", "3",   10 / 1031],
                ["pooled", "a", "f2", "x",   21 / 1031],
                ["pooled", "a", "f2", "y", 1010 / 1031],
                ["pooled", "b", "f1", "2",  102 /  302],
                ["pooled", "b", "f1", "3",  200 /  302],
                ["pooled", "b", "f2", "x",  202 /  302],
                ["pooled", "b", "f2", "y",  100 /  302],
            ],
        )
        actual = count_clonotype_features(
            repertoire=repertoire,
            repertoire_groups=["g1", "g2"],
            clonotype_features=["f1", "f2"],
            clonesize="clonesize",
            partial_repertoire_pools=[[], ["g1"]],
            shared_clonotype_feature_groups=["g1"],
        )
        assert_frame_equal(actual, expected)
