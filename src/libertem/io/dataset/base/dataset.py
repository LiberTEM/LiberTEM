import typing

import numpy as np

from libertem.io.utils import get_partition_shape
from libertem.io.dataset.base import DataSetException
from libertem.web.messageconverter import MessageConverter
from libertem.corrections.corrset import CorrectionSet
from .partition import BasePartition


class DataSet(object):
    def __init__(self, io_backend=None):
        self._cores = 1
        self._sync_offset = 0
        self._sync_offset_info = None
        self._image_count = 0
        self._nav_shape_product = 0
        self._io_backend = io_backend

    def initialize(self, executor):
        """
        Perform possibly expensive initialization, like pre-loading metadata.

        This is run on the master node, but can execute parts on workers, for example
        if they need to access the data stored on worker nodes, using the passed executor
        instance.

        If you need the executor around for later operations, for example when creating
        the partitioning, save a reference here!

        Should return the possibly modified `DataSet` instance (if a method running
        on a worker is changing `self`, these changes won't automatically be transferred back
        to the master node)
        """
        raise NotImplementedError()

    def set_num_cores(self, cores):
        self._cores = cores

    def get_sync_offset_info(self):
        """
        Check sync_offset specified and returns number of frames skipped and inserted
        """
        if not -1*self._image_count < self._sync_offset < self._image_count:
            raise DataSetException(
                "sync_offset should be in (%s, %s), which is (-image_count, image_count)"
                % (-1*self._image_count, self._image_count)
            )
        return {
            "frames_skipped_start": max(0, self._sync_offset),
            "frames_ignored_end": max(
                0, self._image_count - self._nav_shape_product - self._sync_offset
            ),
            "frames_inserted_start": abs(min(0, self._sync_offset)),
            "frames_inserted_end": max(
                0, self._nav_shape_product - self._image_count + self._sync_offset
            )
        }

    def get_num_partitions(self):
        """
        Returns the number of partitions the dataset should be split into
        """
        raise NotImplementedError()

    def get_slices(self):
        """
        Return the partition slices for the dataset
        """
        return BasePartition.make_slices(
            shape=self.shape,
            num_partitions=self.get_num_partitions(),
            sync_offset=self._sync_offset,
        )

    def get_partitions(self):
        """
        Return a generator over all Partitions in this DataSet. Should only
        be called on the master node.
        """
        raise NotImplementedError()

    @property
    def dtype(self):
        """
        the destination data type
        """
        raise NotImplementedError()

    @property
    def raw_dtype(self):
        """
        the underlying data type
        """
        raise NotImplementedError()

    @property
    def shape(self):
        """
        The shape of the DataSet, as it makes sense for the application domain
        (for example, 4D for pixelated STEM)
        """
        raise NotImplementedError()

    def check_valid(self):
        """
        check validity of the DataSet. this will be executed (after initialize) on a worker node.
        should raise DataSetException in case of errors, return True otherwise.
        """
        raise NotImplementedError()

    @classmethod
    def detect_params(cls, path, executor):
        """
        Guess if path can be opened using this DataSet implementation and
        detect parameters.

        returns dict of detected parameters if path matches this dataset type,
        returns False if path is most likely not of a matching type.
        """
        # FIXME: return hints for the user and additional values,
        # for example number of signal elements
        raise NotImplementedError()

    @classmethod
    def get_msg_converter(cls) -> typing.Type[MessageConverter]:
        raise NotImplementedError()

    @property
    def diagnostics(self):
        """
        Diagnostics common for all DataSet implementations
        """
        p = next(self.get_partitions())

        return self.get_diagnostics() + [
            {"name": "Partition shape",
             "value": str(p.shape)},
            {"name": "Number of partitions",
             "value": str(len(list(self.get_partitions())))},
            {"name": "Number of frames skipped at the beginning",
             "value": self._sync_offset_info["frames_skipped_start"]},
            {"name": "Number of frames ignored at the end",
            "value": self._sync_offset_info["frames_ignored_end"]},
            {"name": "Number of blank frames inserted at the beginning",
            "value": self._sync_offset_info["frames_inserted_start"]},
            {"name": "Number of blank frames inserted at the end",
            "value": self._sync_offset_info["frames_inserted_end"]}
        ]

    def get_diagnostics(self):
        """
        Get relevant diagnostics for this dataset, as a list of
        dicts with keys name, value, where value may be string or
        a list of dicts itself. Subclasses should override this method.
        """
        return []

    def partition_shape(self, dtype, target_size, min_num_partitions=None):
        """
        Calculate partition shape for the given ``target_size``

        Parameters
        ----------
        dtype : numpy.dtype or str
            data type of the dataset

        target_size : int
            target size in bytes - how large should each partition be?

        min_num_partitions : int
            minimum number of partitions desired. Defaults to the number of workers in the cluster.

        Returns
        -------
        Tuple[int]
            the shape calculated from the given parameters
        """
        if min_num_partitions is None:
            min_num_partitions = self._cores
        return get_partition_shape(
            dataset_shape=self.shape,
            target_size_items=target_size // np.dtype(dtype).itemsize,
            min_num=min_num_partitions
        )

    @classmethod
    def get_supported_extensions(cls) -> typing.Set[str]:
        """
        Return supported extensions as a set of strings.

        Plain extensions only, no pattern!
        """
        return set()

    def get_cache_key(self):
        raise NotImplementedError()

    def get_io_backend(self):
        return self._io_backend

    def get_correction_data(self):
        """
        Correction parameters that are part of this DataSet.
        This should only be called after the DataSet is initialized.

        Returns
        -------
        CorrectionSet
            correction parameters that are part of this DataSet
        """
        return CorrectionSet()


class WritableDataSet:
    pass
