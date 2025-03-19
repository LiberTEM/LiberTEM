import typing
from typing import Optional
from collections.abc import Generator, Sequence

import numpy as np
from sparseconverter import CUDA, NUMPY, ArrayBackend

from libertem.common.shape import Shape
from libertem.common.math import prod
from libertem.io.utils import get_partition_shape
from libertem.io.dataset.base import DataSetException, MMapBackend
from libertem.common.messageconverter import MessageConverter
from libertem.io.corrections.corrset import CorrectionSet
from .partition import BasePartition, Partition

if typing.TYPE_CHECKING:
    from libertem.common.executor import JobExecutor, TaskCommHandler
    from libertem.io.dataset.base import IOBackend, Decoder, DataSetMeta
    from numpy import typing as nt


class RoiHelper:
    def __init__(self, ds):
        self._ds = ds

    def __getitem__(self, k):
        roi = np.zeros(self._ds.shape.nav, dtype=bool)
        roi[k] = True
        return roi


class DataSet:
    # The default partition size in bytes
    MAX_PARTITION_SIZE = 512*1024*1024

    def __init__(
        self,
        io_backend: Optional["IOBackend"] = None,
        num_partitions: Optional[int] = None,
    ):
        self._cores = 1
        self._sync_offset: Optional[int] = 0
        self._sync_offset_info = None
        self._image_count = 0
        self._nav_shape_product = 0
        self._io_backend = io_backend
        self._user_num_partitions = num_partitions
        self._meta: Optional[DataSetMeta] = None
        self._roi_helper = RoiHelper(ds=self)

    def initialize(self, executor) -> "DataSet":
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

    def set_num_cores(self, cores: int) -> None:
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

    def get_num_partitions(self) -> int:
        """
        Returns the number of partitions the dataset should be split into.

        The default implementation sizes partition such that they
        fit into 512MB of float data in memory, regardless of their
        native dtype. At least :code:`self._cores` partitions
        are created.
        """
        if self._user_num_partitions is not None:
            return max(1, self._user_num_partitions)
        partition_size_float_px = self.MAX_PARTITION_SIZE // 4
        dataset_size_px = prod(self.shape)
        num: int = max(self._cores, dataset_size_px // partition_size_float_px)
        return num

    def get_slices(self):
        """
        Return the partition slices for the dataset
        """
        return BasePartition.make_slices(
            shape=self.shape,
            num_partitions=self.get_num_partitions(),
            sync_offset=self._sync_offset,
        )

    def get_partitions(self) -> Generator[Partition, None, None]:
        """
        Return a generator over all Partitions in this DataSet. Should only
        be called on the master node.
        """
        raise NotImplementedError()

    @property
    def dtype(self) -> "nt.DTypeLike":
        """
        The "native" data type
        (either one matching the data on disk, or one that is closest)
        """
        raise NotImplementedError()

    @property
    def shape(self) -> Shape:
        """
        The shape of the DataSet, as it makes sense for the application domain
        (for example, 4D for pixelated STEM)
        """
        raise NotImplementedError()

    @property
    def array_backends(self) -> Sequence[ArrayBackend]:
        """
        The array backends the dataset can return data as.

        Defaults to only NumPy arrays

        .. versionadded:: 0.11.0
        """
        return (NUMPY, CUDA)

    def check_valid(self) -> bool:
        """
        check validity of the DataSet. this will be executed (after initialize) on a worker node.
        should raise DataSetException in case of errors, return True otherwise.
        """
        raise NotImplementedError()

    @classmethod
    def detect_params(cls, path: str, executor: "JobExecutor"):
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
    def get_msg_converter(cls) -> type[MessageConverter]:
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

    @property
    def roi(self):
        return self._roi_helper

    def partition_shape(
        self,
        dtype: "nt.DTypeLike",
        target_size: int,
        min_num_partitions: Optional[int] = None,
        containing_shape: Optional[Shape] = None,
    ) -> tuple[int, ...]:
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
        Tuple[int, ...]
            the shape calculated from the given parameters
        """
        if min_num_partitions is None:
            min_num_partitions = self._cores
        if containing_shape is None:
            containing_shape = self.shape
        return get_partition_shape(
            dataset_shape=containing_shape,
            target_size_items=target_size // np.dtype(dtype).itemsize,
            min_num=min_num_partitions
        )

    @classmethod
    def get_supported_extensions(cls) -> set[str]:
        """
        Return supported extensions as a set of strings.

        Plain extensions only, no pattern!
        """
        return set()

    def get_cache_key(self) -> str:
        raise NotImplementedError()

    @classmethod
    def get_default_io_backend(cls) -> "IOBackend":
        import platform
        if platform.system() == "Windows":
            from libertem.io.dataset.base import BufferedBackend
            return BufferedBackend()
        return MMapBackend()

    @classmethod
    def get_supported_io_backends(cls) -> list[str]:
        """
        Get the supported I/O backends as list of their IDs. Some DataSet
        implementations with a custom backend may return an empty list here.
        """
        return ["mmap", "buffered", "direct"]

    def get_io_backend(self) -> "IOBackend":
        if self._io_backend is None:
            return self.get_default_io_backend()
        return self._io_backend

    def get_correction_data(self) -> CorrectionSet:
        """
        Correction parameters that are part of this DataSet.
        This should only be called after the DataSet is initialized.

        Returns
        -------
        CorrectionSet
            correction parameters that are part of this DataSet
        """
        return CorrectionSet()

    def supports_correction(self):
        return True

    def get_decoder(self) -> Optional["Decoder"]:
        return None

    def get_base_shape(self, roi: Optional[np.ndarray]) -> tuple[int, ...]:
        return (1,) + (1,) * (self.shape.sig.dims - 1) + (self.shape.sig[-1],)

    def adjust_tileshape(
        self, tileshape: tuple[int, ...], roi: Optional[np.ndarray]
    ) -> tuple[int, ...]:
        """
        Final veto of the DataSet in the tileshape negotiation process,
        make sure that corrections are taken into account!
        """
        return tileshape

    def need_decode(
        self,
        read_dtype: "nt.DTypeLike",
        roi: Optional[np.ndarray],
        corrections: Optional[CorrectionSet],
    ) -> bool:
        io_backend = self.get_io_backend().get_impl()
        return io_backend.need_copy(
            decoder=self.get_decoder(),
            roi=roi,
            native_dtype=self.meta.raw_dtype,
            read_dtype=read_dtype,
            sync_offset=self._sync_offset,
            corrections=corrections,
        )

    def get_min_sig_size(self) -> int:
        """
        minimum signal size, in number of elements
        """
        return 4 * 4096 // np.dtype(self.meta.raw_dtype).itemsize

    def get_max_io_size(self) -> Optional[int]:
        """
        Override this method to implement a custom maximum I/O size (in bytes)
        """
        return None

    @property
    def meta(self) -> Optional["DataSetMeta"]:
        return self._meta

    def get_task_comm_handler(self) -> "TaskCommHandler":
        from libertem.common.executor import NoopCommHandler
        return NoopCommHandler()


class WritableDataSet:
    pass
