"""
Clonotype feature frequencies
-----------------------------

:func:`pymmunomics.preprocessing.repertoire.count_features` coveniently
calculates clonotype feature frequencies (the fractions of immune
repertoire cells that share certain features), such as V/J-gene, or CDR3
length usage frequencies.

In the following example, the frequencies of the different V-genes and
J-genes in the different subject repertoires are calculated:

>>> import pandas as pd
>>> from pymmunomics.preprocessing.repertoire import count_features
>>> 
>>> repertoire = pd.DataFrame(
...     columns=["subject", "Vgene", "Jgene"],
...     data=[
...         # subject1
...         ["subject1", "TRBV6-6", "TRBJ2-7"],
...         ["subject1", "TRBV6-6", "TRBJ2-7"],
...         ["subject1", "TRBV30" , "TRBJ2-7"],
...         ["subject1", "TRBV30" , "TRBJ2-7"],
...         ["subject1", "TRBV30" , "TRBJ2-5"],
...         # subject2
...         ["subject2", "TRBV6-6", "TRBJ4-2"],
...         ["subject2", "TRBV6-6", "TRBJ2-7"],
...         ["subject2", "TRBV9"  , "TRBJ4-2"],
...         ["subject2", "TRBV9"  , "TRBJ2-7"],
...         ["subject2", "TRBV30" , "TRBJ4-2"],
...         ["subject2", "TRBV30" , "TRBJ2-7"],
...     ],
... )
>>> count_features(
...     repertoire=repertoire,
...     repertoire_groups=["subject"],
...     clonotype_features=["Vgene", "Jgene"],
... )
    subject feature feature_value  frequency
0  subject1   Jgene       TRBJ2-5   0.200000
1  subject1   Jgene       TRBJ2-7   0.800000
2  subject1   Vgene        TRBV30   0.600000
3  subject1   Vgene       TRBV6-6   0.400000
4  subject2   Jgene       TRBJ2-7   0.500000
5  subject2   Jgene       TRBJ4-2   0.500000
6  subject2   Vgene        TRBV30   0.333333
7  subject2   Vgene       TRBV6-6   0.333333
8  subject2   Vgene         TRBV9   0.333333

For each of the two subjects ("subject1", "subject2") and for each
chosen clonotype feature ("Vgene" and "Jgene"), the function counts
the rows with the same value for that feature's column and normalizes
by the total number of rows for that subject.

Some basic variations:

- Instead of frequencies one can obtain absolute counts or one-hot
  encodings via the parameter `stat`
- To make sure each subject has counts for the the same feature values
  in the resulting table, one can use the parameter
  `shared_clonotype_feature_groups`
- For partially aggregated input data providing clonotypes and their
  counts, the parameter `clonesize` should be used

>>> import pandas as pd
>>> from pymmunomics.preprocessing.repertoire import count_features
>>> 
>>> repertoire = pd.DataFrame(
...     columns=["subject", "Vgene", "Jgene", "clone_size"],
...     data=[
...         # subject1
...         ["subject1", "TRBV6-6", "TRBJ2-7", 2],
...         ["subject1", "TRBV30" , "TRBJ2-7", 2],
...         ["subject1", "TRBV30" , "TRBJ2-5", 1],
...         # subject2
...         ["subject2", "TRBV6-6", "TRBJ4-2", 1],
...         ["subject2", "TRBV6-6", "TRBJ2-7", 10],
...         ["subject2", "TRBV9"  , "TRBJ4-2", 1],
...         ["subject2", "TRBV9"  , "TRBJ2-7", 10],
...         ["subject2", "TRBV30" , "TRBJ4-2", 1],
...         ["subject2", "TRBV30" , "TRBJ2-7", 10],
...     ],
... )
>>> count_features(
...     repertoire=repertoire,
...     repertoire_groups=["subject"],
...     clonotype_features=["Vgene", "Jgene"],
...     clonesize="clone_size",
...     stat="count",
...     shared_clonotype_feature_groups=["subject"],
... )
     subject feature feature_value  count
0   subject1   Jgene       TRBJ2-5      1
1   subject1   Jgene       TRBJ2-7      4
2   subject1   Jgene       TRBJ4-2      0
3   subject1   Vgene        TRBV30      3
4   subject1   Vgene       TRBV6-6      2
5   subject1   Vgene         TRBV9      0
6   subject2   Jgene       TRBJ2-5      0
7   subject2   Jgene       TRBJ2-7     30
8   subject2   Jgene       TRBJ4-2      3
9   subject2   Vgene        TRBV30     11
10  subject2   Vgene       TRBV6-6     11
11  subject2   Vgene         TRBV9     11

In a more complex experiment, clonotypes may be grouped in different
ways, such as into different cell types and cell subtypes. This can be
achieved via the parameters `repertoire_groups` and
`partial_repertoire_pools`.

>>> import pandas as pd
>>> from pymmunomics.preprocessing.repertoire import count_features
>>> 
>>> repertoire = pd.DataFrame(
...     columns=["cell type", "cell subtype", "subject", "Vgene", "Jgene", "clone_size"],
...     data=[
...         # TRB subject1
...         ["TRB", "CD4", "subject1", "TRBV6-6" , "TRBJ2-7", 2],
...         ["TRB", "CD4", "subject1", "TRBV30"  , "TRBJ2-7", 2],
...         ["TRB", "CD8", "subject1", "TRBV30"  , "TRBJ2-5", 1],
...         # TRB subject2
...         ["TRB", "CD4", "subject2", "TRBV6-6" , "TRBJ4-2", 1],
...         ["TRB", "CD4", "subject2", "TRBV6-6" , "TRBJ2-7", 10],
...         ["TRB", "CD4", "subject2", "TRBV9"   , "TRBJ4-2", 1],
...         ["TRB", "CD8", "subject2", "TRBV9"   , "TRBJ2-7", 10],
...         ["TRB", "CD8", "subject2", "TRBV30"  , "TRBJ4-2", 1],
...         ["TRB", "CD8", "subject2", "TRBV30"  , "TRBJ2-7", 10],
...         # IGH subject1
...         ["IGH", "IgG", "subject1", "IGHV3-33", "IGHJ2"  , 2],
...         ["IGH", "IgM", "subject1", "IGHV1-18", "IGHJ2"  , 2],
...         ["IGH", "IgM", "subject1", "IGHV1-18", "IGHJ5"  , 1],
...         # IGH subject2
...         ["IGH", "IgG", "subject2", "IGHV3-33", "IGHJ4"  , 1],
...         ["IGH", "IgG", "subject2", "IGHV3-33", "IGHJ2"  , 10],
...         ["IGH", "IgM", "subject2", "IGHV4-4" , "IGHJ4"  , 1],
...         ["IGH", "IgM", "subject2", "IGHV4-4" , "IGHJ2"  , 10],
...         ["IGH", "IgM", "subject2", "IGHV1-18", "IGHJ4"  , 1],
...         ["IGH", "IgM", "subject2", "IGHV1-18", "IGHJ2"  , 10],
...     ],
... )
>>> count_features(
...     repertoire=repertoire,
...     repertoire_groups=["cell type", "cell subtype", "subject"],
...     clonotype_features=["Vgene", "Jgene"],
...     clonesize="clone_size",
...     partial_repertoire_pools=[[], ["cell subtype"]],
...     stat="count",
...     shared_clonotype_feature_groups=["subject"],
... )
   cell type cell subtype   subject feature feature_value  count
0        IGH          IgG  subject1   Jgene         IGHJ2      2
1        IGH          IgG  subject1   Jgene         IGHJ4      0
2        IGH          IgG  subject1   Vgene      IGHV3-33      2
3        IGH          IgG  subject2   Jgene         IGHJ2     10
4        IGH          IgG  subject2   Jgene         IGHJ4      1
5        IGH          IgG  subject2   Vgene      IGHV3-33     11
6        IGH          IgM  subject1   Jgene         IGHJ2      2
7        IGH          IgM  subject1   Jgene         IGHJ4      0
8        IGH          IgM  subject1   Jgene         IGHJ5      1
9        IGH          IgM  subject1   Vgene      IGHV1-18      3
10       IGH          IgM  subject1   Vgene       IGHV4-4      0
11       IGH          IgM  subject2   Jgene         IGHJ2     20
12       IGH          IgM  subject2   Jgene         IGHJ4      2
13       IGH          IgM  subject2   Jgene         IGHJ5      0
14       IGH          IgM  subject2   Vgene      IGHV1-18     11
15       IGH          IgM  subject2   Vgene       IGHV4-4     11
16       IGH       pooled  subject1   Jgene         IGHJ2      4
17       IGH       pooled  subject1   Jgene         IGHJ4      0
18       IGH       pooled  subject1   Jgene         IGHJ5      1
19       IGH       pooled  subject1   Vgene      IGHV1-18      3
20       IGH       pooled  subject1   Vgene      IGHV3-33      2
21       IGH       pooled  subject1   Vgene       IGHV4-4      0
22       IGH       pooled  subject2   Jgene         IGHJ2     30
23       IGH       pooled  subject2   Jgene         IGHJ4      3
24       IGH       pooled  subject2   Jgene         IGHJ5      0
25       IGH       pooled  subject2   Vgene      IGHV1-18     11
26       IGH       pooled  subject2   Vgene      IGHV3-33     11
27       IGH       pooled  subject2   Vgene       IGHV4-4     11
28       TRB          CD4  subject1   Jgene       TRBJ2-7      4
29       TRB          CD4  subject1   Jgene       TRBJ4-2      0
30       TRB          CD4  subject1   Vgene        TRBV30      2
31       TRB          CD4  subject1   Vgene       TRBV6-6      2
32       TRB          CD4  subject1   Vgene         TRBV9      0
33       TRB          CD4  subject2   Jgene       TRBJ2-7     10
34       TRB          CD4  subject2   Jgene       TRBJ4-2      2
35       TRB          CD4  subject2   Vgene        TRBV30      0
36       TRB          CD4  subject2   Vgene       TRBV6-6     11
37       TRB          CD4  subject2   Vgene         TRBV9      1
38       TRB          CD8  subject1   Jgene       TRBJ2-5      1
39       TRB          CD8  subject1   Jgene       TRBJ2-7      0
40       TRB          CD8  subject1   Jgene       TRBJ4-2      0
41       TRB          CD8  subject1   Vgene        TRBV30      1
42       TRB          CD8  subject1   Vgene         TRBV9      0
43       TRB          CD8  subject2   Jgene       TRBJ2-5      0
44       TRB          CD8  subject2   Jgene       TRBJ2-7     20
45       TRB          CD8  subject2   Jgene       TRBJ4-2      1
46       TRB          CD8  subject2   Vgene        TRBV30     11
47       TRB          CD8  subject2   Vgene         TRBV9     10
48       TRB       pooled  subject1   Jgene       TRBJ2-5      1
49       TRB       pooled  subject1   Jgene       TRBJ2-7      4
50       TRB       pooled  subject1   Jgene       TRBJ4-2      0
51       TRB       pooled  subject1   Vgene        TRBV30      3
52       TRB       pooled  subject1   Vgene       TRBV6-6      2
53       TRB       pooled  subject1   Vgene         TRBV9      0
54       TRB       pooled  subject2   Jgene       TRBJ2-5      0
55       TRB       pooled  subject2   Jgene       TRBJ2-7     30
56       TRB       pooled  subject2   Jgene       TRBJ4-2      3
57       TRB       pooled  subject2   Vgene        TRBV30     11
58       TRB       pooled  subject2   Vgene       TRBV6-6     11
59       TRB       pooled  subject2   Vgene         TRBV9     11

The partial pooling achieved one round of counts grouped by all of the
grouping variables `["cell type", "cell subtype", "subject"]` which can
be thought of as partially pooled by columns `[]` (the first item in the
argument of `partial_repertoire_pools`) and one round of counts where
clonotypes from different subtypes are pooled together (corresponding to
the second item in the argument `["cell subtype"]`
`partial_repertoire_pools`)

# CDR3 length comparison

Binding capacity
----------------

`pymmunomics` faciliates the calculation of binding similarities between
pairs of clonotypes.

The binding similarity of a pair of clonotypes is 0
when V- or J-genes disagree and otherwise a function of the Levenshtein
distance between the two CDR3 sequences. The exact function was
determined experimentally to be `0.3 ** Levenshtein_distance`. To
calculate binding similarity, use
:func:`pymmunomics.sim.similarity.binding_similarity`.

>>> from pymmunomics.sim.similarity import binding_similarity
>>> 
>>> left = ("CARDYW", "IGHV1-3", "IGHJ4")
>>> for right in [
...     ("CARWWWDYW", "IGHV1-3", "IGHJ4"), # same V-/J-genes: similarity == levenshtein(CARDYW, CARWWWDYW)
...     ("CARWWWDYW", "IGHV1-46", "IGHJ4"), # different V-genes: similarity == 0
...     ("CARWWWDYW", "IGHV1-3", "IGHJ2"), # different J-genes: similarity == 0
...     left, # same clonotype: similarity == 1
... ]: print(binding_similarity(left=left, right=right))
0.026999999999999996
0.0
0.0
1.0

The input types for the two clonotypes can be any that allows for random
access and contains CDR3 sequence, V-gene, and J-gene at coordinates 0,
1, and 2, respectively.

Average similarity between a query set of clonotypes and a repertoire of
clonotypes (a repertoire being a set of clonotypes with frequencies) can
be measured via the `weighted_similarities` method of any of the classes
`SimilarityFromDataFrame`, `SimilarityFromArray`, `SimilarityFromFile`,
`SimilarityFromFunction` in :mod:`pymmunomics.sim.similarity`.
The choice of class is driven by the way similarities are calculated,
with each class either relying on a precalculated similarity matrix
stored in a pandas.DataFrame, a numpy.ndarray, or in a file, or
calculating similarities on the fly from a python function.
:func:`pymmunomics.sim.similarity.make_similarity` can be used to
conveninently create the right object for calculating average
similarities.

To calculate binding capacities without precomputed similarity matrix,
:func:`pymmunomics.sim.similarity.SimilarityFromFunction` can be used
with :func:`pymmunomics.sim.similarity.binding_similarity` as its
similarity function.

>>> import pandas as pd
>>> from pymmunomics.sim.similarity import binding_similarity, make_similarity
>>> from pymmunomics.helper.log import set_log_level
>>> 
>>> set_log_level("DEBUG") # avoid some log statements from being printed
>>> 
>>> repertoire = pd.DataFrame(
...     columns=["cdr3_aa", "v_gene", "j_gene", "frequency"],
...     data=[
...         ["CARDYW"   , "IGHV1-3", "IGHJ4", 0.1],
...         ["CARWDYW"  , "IGHV1-3", "IGHJ4", 0.2],
...         ["CARWWDYW" , "IGHV1-3", "IGHJ4", 0.3],
...         ["CARWWWDYW", "IGHV1-3", "IGHJ4", 0.2],
...         ["CARWWWDFW", "IGHV1-3", "IGHJ2", 0.2],
...     ],
... )
>>> 
>>> query_clonotypes = pd.DataFrame(
...     columns=["cdr3_aa", "v_gene", "j_gene"],
...     data=[
...         ["CARDYW"   , "IGHV1-3" , "IGHJ4"], # matches first 4 repertoire clonotypes
...         ["CARWWWDYW", "IGHV1-3" , "IGHJ4"], # matches first 4 repertoire clonotypes
...         ["CARDFW"   , "IGHV1-3" , "IGHJ2"], # matches 5th repertoire clonotype
...         ["CARDYW"   , "IGHV1-46", "IGHJ4"], # no match due to V-gene
...         ["CARDYW"   , "IGHV1-3" , "IGHJ3"], # no match due to J-gene
...     ],
... )
>>> 
>>> sim = make_similarity(
...     similarity=binding_similarity,
...     X=query_clonotypes.to_numpy(),
...     Y=repertoire[["cdr3_aa", "v_gene", "j_gene"]].to_numpy(),
... )
>>> 
>>> sim.weighted_similarities(
...     species_frequencies=repertoire[["frequency"]].to_numpy(),
... )
array([[0.1924],
       [0.3107],
       [0.0054],
       [0.    ],
       [0.    ]])

- the :func:`pymmunomics.sim.similarity.SimilarityFromFunction.weighted_similarities`
  is parallelized
- the argument `similarities_out` for :func:`pymmunomics.sim.similarity.make_similarity`
  can be used to store the calculated similarities in a numpy.ndarray-like
  structure
- the argument `chunk_size` for :func:`pymmunomics.sim.similarity.make_similarity`
  can be tuned to accelerate computation
- If the argument `Y` for :func:`pymmunomics.sim.similarity.make_similarity`
  is None, similarities are calculated between all pairs of clonotypes
  in `X`

Feature selection
-----------------

`pymmunomics` offers a number of related feature selection mechanisms
that are designed to integrate well with the scikit-learn API by
inheriting from sklearn.base.TransformerMixin.

The underlying philosophy of these feature selection mechanisms is that
domain knowledge should be preferred over generic feature selection
mechanisms. To that end, the algorithms implemented by `pymmunomics`
require a scoring function for features and access to a set of
feature measurements whose score represent a background or null
distribution against which to compare the scores of features from which
we select.

For example, non-productively recombined germline receptor sequence
repertoires are often used as a null model for signals in the
productively recombined receptor sequence repertoires arising from
disease-specific selection. Measuring correlation of V-gene frequencies
with the response variable of interest would be a way of scoring V-gene
frequency features. 

:class:`pymmunomics.ml.feature_selection.SelectPairedNullScoreOutlier`
selects features in this manner when the null features are paired
measurements to the features from which we select. More precisely, the
mechanism calculates scores (by default Kendall-Tau C correlation
coefficient) from the features together with the response variable.
Outliers in the spread of differences of train-data scores minus
null-data scores indicate which features should be selected.

>>> import numpy as np
>>> import pandas as pd
>>> from pymmunomics.helper.log import set_log_level
>>> from pymmunomics.ml.feature_selection import SelectPairedNullScoreOutlier
>>> 
>>> set_log_level("DEBUG") # avoid some log statements from being printed
>>> 
>>> train_X = pd.DataFrame(
...     columns=["IGHV1-18", "IGHV3-7", "IGHV3-11", "IGHV3-30", "IGHV4-4", "IGHV4-34"],
...     index=["case1", "case2", "case3", "case4", "case5", "control6", "control7", "control8", "control9"],
...     data=[
...         [0.10, 0.20, 0.30, 0.20, 0.10, 0.10],
...         [0.11, 0.21, 0.28, 0.20, 0.09, 0.11],
...         [0.09, 0.20, 0.29, 0.22, 0.11, 0.09],
...         [0.10, 0.20, 0.33, 0.19, 0.08, 0.10],
...         [0.08, 0.19, 0.31, 0.21, 0.11, 0.10],
...         [0.10, 0.18, 0.26, 0.19, 0.08, 0.19],
...         [0.08, 0.19, 0.30, 0.17, 0.06, 0.20],
...         [0.06, 0.17, 0.30, 0.19, 0.08, 0.20],
...         [0.06, 0.17, 0.27, 0.21, 0.09, 0.20],
...     ]
... )
>>> train_y = np.array([0.41, 0.35, 0.58, 0.45, 0.44, 0.21, 0.23, 0.30, 0.18])
>>> null_X = pd.DataFrame(
...     columns=["IGHV1-18", "IGHV3-7", "IGHV3-11", "IGHV3-30", "IGHV4-4", "IGHV4-34"],
...     index=["case1", "case2", "case3", "case4", "case5", "control6", "control7", "control8", "control9"],
...     data=[
...         [0.08, 0.18, 0.28, 0.18, 0.08, 0.20],
...         [0.09, 0.19, 0.26, 0.18, 0.07, 0.21],
...         [0.07, 0.18, 0.27, 0.20, 0.09, 0.19],
...         [0.08, 0.18, 0.31, 0.17, 0.06, 0.20],
...         [0.06, 0.17, 0.29, 0.19, 0.09, 0.20],
...         [0.10, 0.18, 0.26, 0.19, 0.08, 0.19],
...         [0.08, 0.19, 0.30, 0.17, 0.06, 0.20],
...         [0.06, 0.17, 0.30, 0.19, 0.08, 0.20],
...         [0.06, 0.17, 0.27, 0.21, 0.09, 0.20],
...     ]
... )
>>> null_y = np.array([0.41, 0.35, 0.58, 0.45, 0.44, 0.21, 0.23, 0.30, 0.18])
>>> selector = SelectPairedNullScoreOutlier(
...     null_X=null_X,
...     null_y=null_y,
...     alpha=0.5, # select everything with score 
... ).fit(train_X, train_y)
>>> print(f"train_scores - null_scores: {selector.train_scores - selector.null_scores}")
>>> print(f"lower and upper quantiles: {selector.lower_quantile}, {selector.upper_quantile}")
>>> print(f"selected columns: {selector.selected_columns}")
>>> selector.transform(train_X)
train_scores - null_scores: [ 0.33950617  0.45061728  0.28230453  0.49382716  0.43004115 -0.72839506]
lower and upper quantiles: 0.23176954732510288, 0.4527777777777777
selected columns: Index(['IGHV3-30', 'IGHV4-34'], dtype='object')
          IGHV3-30  IGHV4-34
case1         0.20      0.10
case2         0.20      0.11
case3         0.22      0.09
case4         0.19      0.10
case5         0.21      0.10
control6      0.19      0.19
control7      0.17      0.20
control8      0.19      0.20
control9      0.21      0.20

The data from which to calculate train scores can also be given in the
initializer of the object using the parameters `train_X` and `train_y`,
which can be helpful for example when a continuous variable is used
during scoring and its binned values are used during actual training and
classificiation of the model.

:class:`pymmunomics.ml.feature_selection.SelectNullScoreOutlier` is
meant to select features when the null features are not paired
measurements. In this mechanism, scores are calculated from all features
(train and null) together with the response variable. train features
whose score falls outside the desired quantiles of the null feature
scores are selected.
"""
