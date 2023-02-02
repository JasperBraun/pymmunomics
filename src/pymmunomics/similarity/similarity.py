"""Module for calculating (weighted) similarities.

Classes
-------
Similarity
    Abstract base class for calculating (weighted) similarities.
SimilarityFromDataFrame
    Implements Similarity using a precomputed similarity matrix stored
    in a pandas.DataFrame.
SimilarityFromArray
    Implements Similarity using a precomputed similarity matrix stored
    in a numpy.ndarray or numpy.memmap.
SimilarityFromFile
    Implements Similarity by reading similarities from a csv or tsv 
    file in chunks.
SimilarityFromFunction:
    Implements Similarity by calculating pairwise similarities from a 
    user-specified callable.
"""
from abc import ABC, abstractmethod
from typing import Callable, Union

from numpy import memmap, ndarray, empty, concatenate, float64
from numpy.typing import ArrayLike
from pandas import DataFrame, Index, MultiIndex, read_csv
from polyleven import levenshtein
from scipy.sparse import spmatrix
from ray import remote, get, put

from pymmunomics.helper.log import LOGGER

def binding_similarity(left, right):
    if (left[1], left[2]) != (right[1], right[2]):
        return 0.0
    else:
        return 0.3**levenshtein(left[0], right[0])

def expected_similarities(
    similarity_function: Callable,
    query: ndarray,
    query_index: Union[Index, MultiIndex],
    subject: ndarray,
    subject_relative_abundance: ndarray,
    similarity_matrix_filepath: Union[str, None]=None,
) -> DataFrame:
    """Calculates expected similarities of query to subject species.

    Parameters
    ----------
    similarity_function:
        Returns similarity between a pair of query and subject species.
    query, subject:
        Contain query and subject species. `similarity_function` is
        applied to each pair of query and subject rows.
    query_index:
        Indexes query species and is used as index of the returned
        expected similarity table.
    subject_relative_abundance:
        (1, n) array of query species frequencies (n is the number of
        species in `query`).
    similarity_matrix_filepath:
        Path to file to store binary similarity matrix, if desired.

    Returns
    -------
    expected_similarities_table:
        The expected similarities of each query species to the subject
        species.
    """
    if similarity_matrix_filepath is None:
        out = None
    else:
        out = memmap(
            similarity_matrix_filepath,
            dtype=float64,
            mode="w+",
            offset=0,
            shape=(query.shape[0], subject.shape[0]),
            order="C",
        )
    similarity = SimilarityFromFunction(
        similarity=similarity_function,
        X=query,
        Y=subject,
        out=out,
    )
    expected_similarities = similarity.weighted_similarities(
        relative_abundance=subject_relative_abundance,
    )
    expected_similarity_table = (
        DataFrame(
            data=expected_similarities,
            columns=["binding_capacity"],
        )
        .set_index(query_index)
    )
    return expected_similarity_table

class Similarity(ABC):
    """Interface for classes computing (weighted) similarities."""

    def __init__(self, similarity: Union[DataFrame, ndarray, str, Callable]) -> None:
        """
        Parameters
        ----------
        similarity:
            A similarity matrix, a path to a similarity matrix, or a
            function that can be called on pairs of species to calculate
            similarities.
        """
        self.similarity = similarity

    @abstractmethod
    def weighted_similarities(
        self, relative_abundance: Union[ndarray, spmatrix]
    ) -> ndarray:
        """Calculates weighted sums of similarities for each species.

        Parameters
        ----------
        relative_abundance:
            Contains the frequencies for each species (rows) in each
            subcommunity (columns).

        Returns
        -------
        weighted_similarity_sums:
            A 2-d array of the same shape as `relative_abundance`, where
            rows correspond to similarity matrix rows, columns
            correspond to columns in `relative_abundance`, and each
            element is a sum of the similarity matrix row weighted by
            the column in `relative_abundance`.
        """
        pass


class SimilarityFromDataFrame(Similarity):
    """Implements Similarity using similarities stored in pandas
    dataframe."""

    def weighted_similarities(
        self, relative_abundance: Union[ndarray, spmatrix]
    ) -> ndarray:
        return self.similarity.to_numpy() @ relative_abundance


class SimilarityFromArray(Similarity):
    """Implements Similarity using similarities stored in a numpy
    ndarray."""

    def weighted_similarities(
        self, relative_abundance: Union[ndarray, spmatrix]
    ) -> ndarray:
        return self.similarity @ relative_abundance


class SimilarityFromFile(Similarity):
    """Implements Similarity by using similarities stored in file.

    Similarity matrix rows are read from the file one chunk at a time.
    The size of chunks can be specified in numbers of rows to control
    memory load.
    """

    def __init__(self, similarity: str, chunk_size: int = 100) -> None:
        """
        Parameters
        ----------
        similarity:
            Path to a file containing a similarity matrix. The file
            should have a single-row header.
        chunk_size:
            Number of rows to read from similarity matrix at a time.
        """
        super().__init__(similarity=similarity)
        self.chunk_size = chunk_size
        self.n_similarity_rows = read_csv(
            self.similarity,
            sep=None,
            engine="python",
            dtype=float64,
            usecols=[0],
        ).shape[0]

    def weighted_similarities(
        self, relative_abundance: Union[ndarray, spmatrix]
    ) -> ndarray:
        weighted_similarities = empty(
            shape=(self.n_similarity_rows, relative_abundance.shape[1]),
            dtype=float64,
        )
        with read_csv(
            self.similarity,
            chunksize=self.chunk_size,
            sep=None,
            engine="python",
            dtype=float64,
        ) as similarity_matrix_chunks:
            i = 0
            for chunk in similarity_matrix_chunks:
                weighted_similarities[i : i + self.chunk_size, :] = (
                    chunk.to_numpy() @ relative_abundance
                )
                i += self.chunk_size
        return weighted_similarities


@remote
def weighted_similarity_chunk(
    similarity: Callable,
    X: ndarray,
    relative_abundance: ndarray,
    chunk_size: int,
    chunk_index: int,
    Y: ndarray = None,
) -> ndarray:
    if Y is None:
        Y = X
    chunk = X[chunk_index : chunk_index + chunk_size]
    similarities_chunk = empty(shape=(chunk.shape[0], Y.shape[0]))
    for i, row_i in enumerate(chunk):
        for j, row_j in enumerate(Y):
            similarities_chunk[i, j] = similarity(row_i, row_j)
    return (similarities_chunk @ relative_abundance, similarities_chunk)


class SimilarityFromFunction(Similarity):
    """Calculates similarities from user-specified similarity function."""

    def __init__(
        self,
        similarity: Callable,
        X: ndarray,
        Y: ndarray = None,
        out: ndarray = None,
        chunk_size: int = 100,
    ) -> None:
        """
        similarity:
            A Callable that calculates similarity between a pair of
            species. Must take two rows from X, or a row from X and one
            from Y as its arguments and return a numeric similarity
            value.
        X, Y:
            Each row contains the feature values for a given species.
            Rows in X correspond to rows in the calcluated similarity
            matrix (and therefore to rows in the weighted similarities).
            If `Y` is not specified, rows in X also correspond to
            columns in the calculated similarity matrix.
        out:
            If specified, similarity matrix will be stored in this
            array.
        chunk_size:
            Determines how many rows of the similarity matrix will be
            stored in memory at a time. In general, choosing a larger
            `chunk_size` will make the calculation faster, but will also
            require more memory.
        """
        super().__init__(similarity=similarity)
        self.X = X
        self.Y = Y
        self.out = out
        self.chunk_size = chunk_size

    def weighted_similarities(
        self, relative_abundance: Union[ndarray, spmatrix]
    ) -> ndarray:
        X_ref = put(self.X)
        Y_ref = put(self.Y)
        abundance_ref = put(relative_abundance)
        futures = []
        similarities_chunk_futures = []
        for chunk_index in range(0, self.X.shape[0], self.chunk_size):
            chunk_future = weighted_similarity_chunk.remote(
                similarity=self.similarity,
                X=X_ref,
                relative_abundance=abundance_ref,
                chunk_size=self.chunk_size,
                chunk_index=chunk_index,
                Y=Y_ref,
            )
            futures.append(chunk_future)
        weighted_similarity_chunks = get(futures)
        if self.out is not None:
            for chunk_index, similarities_chunk in zip(
                range(0, self.X.shape[0], self.chunk_size),
                weighted_similarity_chunks,
            ):
                self.out[chunk_index:chunk_index+similarities_chunk[1].shape[0]] = similarities_chunk[1]
        return concatenate([
            similarities_chunk[0]
            for similarities_chunk
            in weighted_similarity_chunks
        ])

