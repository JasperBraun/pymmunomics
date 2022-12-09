from functools import partial
from multiprocessing import cpu_count, Pool
from polyleven import levenshtein
from tempfile import TemporaryDirectory
from typing import Union

from numpy import (
    dtype,
    memmap,
    fill_diagonal,
    tril_indices,
)
from pandas import DataFrame, Series

from pymmunomics.helper.generic_helpers import (
    concatenate_files,
    partition_range,
)
from pymmunomics.helper.numpy_helpers import (
    apply_to_memmapped,
    MemMapSpec,
)
from pymmunomics.helper.pandas_helpers import pairwise_apply

def binding_similarity(
    left: Series,
    right: Series,
    seq_col: str,
    match_cols: tuple[str] = tuple(),
    set_intersection_cols: tuple[str] = tuple(),
    base: float = 0.3,
):
    """Calculates binding similarity.

    Binding similarity between CDR3 amino acid sequences left, right can
    be measured as:

        base ** Levenshtein(left, right)

    where base can be obtained empirically from binding studies. To
    allow additional categorical matching, such as for VJ-genes, one can
    define mismatching categories to lead to a similarity of 0. For a
    more lenient treatment, one can gather sets of categories (ex.
    VJ-genes within a certain distance of the cell's VJ-genes) and
    condition non-zero similarity on non-empty intersections of these
    sets of categories.

    Parameters
    ----------
    left, right:
        Cells to calculate similarity for. Must have as index the names
        of columns given by seq_col, match_cols, and
        set_intersection_cols.
    seq_col:
        The index of left and right that indicates the cell's CDR3 amino
        acid sequence.
    match_cols:
        Indexes of left and right that indicate categorical columns
        which require to be matched for non-0 similarity (ex. VJ-genes).
    set_intersection_cols:
        Indexes of left and right that indicate columns containing sets
        of categories which require non-0 intersections to be matched
        for non-0 similarity (ex. VJ-gene neighborhoods).
    base:
        The base to use for the similarity measure.

    Returns
    -------
    The float value of the similarity for the clones described by left
    and right.
    """
    if len(match_cols) > 0:
        for col in match_cols:
            if left[col] != right[col]:
                return 0.0
    if len(set_intersection_cols) > 0:
        for col in set_intersection_cols:
            if len(left[col].intersection(right[col])) == 0:
                return 0.0
    # return base ** distance(left[seq_col], right[seq_col])
    return base ** levenshtein(left[seq_col], right[seq_col])

def calculate_similarity_matrix(
    query: DataFrame,
    similarities_filepath: str,
    subject: Union[DataFrame, None] = None,
    seq_col: str = "cdr3_aa",
    base: float = 0.3,
    binding_similarity_kwargs: Union[dict, None] = None,
    num_cores: Union[int, None] = None,
):
    """Calculates similarities between query and subject clonotypes.

    Parameters
    ----------
    query:
        Table containing query clonotypes. Rows are passed to
        ``pymmunomics.similarity.binding_similarity.binding_similarity``
        as argument for parameter `left`. Columns must satisfy
        configuration required by
        ``pymmunomics.similarity.binding_similarity.binding_similarity``.
    similarities_filepath:
        Path to file in which to store similarity matrix. Will be stored
        as a .npy binary ``numpy.ndarray`` file with dtype
        ``numpy.float64``. Rows correspond to query clonotypes in the
        order of appearance in `query, and columns correspond to subject
        clones in the order of appearance in subject.
    subject:
        Table containing subject clones. Rows are passed to
        ``pymmunomics.similarity.binding_similarity.binding_similarity``
        as argument for parameter `right`. Columns must satisfy
        configuration required by
        ``pymmunomics.similarity.binding_similarity.binding_similarity``.
        Must have additional column `frequency_column. If ``None``,
        `query` is `subject`, and matrix is assumed to be symmetric and
        diagonal == 1.
    seq_col:
        Name of column containing clonotype sequences. Expected to be
        the same for `query` and `subject`.
    base
        Base of binding similarity measure. See
        ``pymmunomics.similarity.binding_similarity.binding_similarity``.
    binding_similarity_kwargs
        Keyword arguments passed to
        ``pymmunomics.similarity.binding_similarity.binding_similarity``.
        Keywords `"base"` and `"seq_col"` override the `base` and
        `seq_col` arguments passed to this function.
    num_cores
        Number of cores to use. If unspecified, uses as many cores as
        possible.
    """
    if subject is None:
        subject = query
        symmetric = True
    else:
        symmetric = False
    if num_cores is None:
        num_cores_ = min(query.shape[0], cpu_count())
    else:
        num_cores_ = min(query.shape[0], num_cores)
    if binding_similarity_kwargs is None:
        binding_similarity_kwargs = {
            "seq_col": seq_col,
            "base": base,
        }
    else:
        for key, value in [("base", base), ("seq_col", seq_col)]:
            if key not in binding_similarity_kwargs:
                binding_similarity_kwargs[key] = value

    query_row_ranges = partition_range(range(query.shape[0]), num_cores_)

    with TemporaryDirectory() as tmp_dir:
        similarity_matrix_chunk_filepaths = [
            f"{tmp_dir}/similarities_{i}.npy" for i in range(num_cores_)
        ]
        combined_similarity_matrix_filepath = similarities_filepath

        apply_to_memmapped_args = [
            (
                pairwise_apply,
                {
                    "out": (
                        similarity_matrix_chunk_filepath,
                        MemMapSpec(
                            dtype=dtype("f8"),
                            mode="w+",
                            offset=0,
                            shape=(len(row_range), subject.shape[0]),
                            order="C",
                        ),
                    ),
                },
                {
                    "func": binding_similarity,
                    "left": query.iloc[row_range],
                    "right": subject,
                    "func_kwargs": binding_similarity_kwargs,
                    "is_valid_relative_position": (
                        partial(_is_valid_position, offset=row_range.start)
                        if symmetric
                        else None
                    ),
                },
            )
            for similarity_matrix_chunk_filepath, row_range in zip(
                similarity_matrix_chunk_filepaths, query_row_ranges
            )
        ]
        with Pool(num_cores_) as pool:
            pool.starmap(apply_to_memmapped, apply_to_memmapped_args)
        concatenate_files(
            input_filepaths=similarity_matrix_chunk_filepaths,
            output_filepath=combined_similarity_matrix_filepath,
            read_mode="rb",
            write_mode="wb",
            delete_input_files=True,
        )

        if symmetric:
            similarity_matrix = memmap(
                combined_similarity_matrix_filepath,
                dtype=dtype("f8"),
                mode="r+",
                offset=0,
                shape=(query.shape[0], subject.shape[0]),
                order="C",
            )
            fill_diagonal(similarity_matrix, 1)
            i_lower = tril_indices(similarity_matrix.shape[0], -1)
            similarity_matrix[i_lower] = similarity_matrix.T[i_lower]

def _is_valid_position(i, j, offset):
    return i + offset < j