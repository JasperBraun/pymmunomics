from numpy import array

from pymmunomics.similarity.similarity import binding_similarity

class TestBindingSimilarity:
    def test_both_genes_differ(self):
        actual = binding_similarity(
            array(["foo", "bar", "baz"]),
            array(["foo", "bar2", "baz2"]),
        )
        expected = 0.0
        assert actual == expected

    def test_first_gene_differs(self):
        actual = binding_similarity(
            array(["foo", "bar", "baz"]),
            array(["foo", "bar2", "baz"]),
        )
        expected = 0.0
        assert actual == expected

    def test_second_gene_differs(self):
        actual = binding_similarity(
            array(["foo", "bar", "baz"]),
            array(["foo", "bar", "baz2"]),
        )
        expected = 0.0
        assert actual == expected

    def test_both_genes_same(self):
        actual = binding_similarity(
            array(["fofo", "bar", "baz"]),
            array(["ofof", "bar", "baz"]),
        )
        expected = 0.3**2
        assert actual == expected

    def test_identical(self):
        actual = binding_similarity(
            array(["foo", "bar", "baz"]),
            array(["foo", "bar", "baz"]),
        )
        expected = 1.0
        assert actual == expected
