import typing
from typing import Union, Any
import pathlib
from functools import lru_cache
import importlib
import warnings
from typing_extensions import Literal
import numpy as np

from libertem.io.dataset.base import DataSetException, DataSet
from libertem.common.async_utils import sync_to_async
from libertem.common.scheduler import Scheduler


filetypes = {
    "hdf5": "libertem.io.dataset.hdf5.H5DataSet",
    "raw": "libertem.io.dataset.raw.RawFileDataSet",
    "raw_csr": "libertem.io.dataset.raw_csr.RawCSRDataSet",
    "mib": "libertem.io.dataset.mib.MIBDataSet",
    "blo": "libertem.io.dataset.blo.BloDataSet",
    "k2is": "libertem.io.dataset.k2is.K2ISDataSet",
    "ser": "libertem.io.dataset.ser.SERDataSet",
    "frms6": "libertem.io.dataset.frms6.FRMS6DataSet",
    "empad": "libertem.io.dataset.empad.EMPADDataSet",
    "dm": "libertem.io.dataset.dm.DMDataSet",
    "seq": "libertem.io.dataset.seq.SEQDataSet",
    "mrc": "libertem.io.dataset.mrc.MRCDataSet",
    "tvips": "libertem.io.dataset.tvips.TVIPSDataSet",
    "npy": "libertem.io.dataset.npy.NPYDataSet",
    "dask": "libertem.io.dataset.dask.DaskDataSet",
    "memory": "libertem.io.dataset.memory.MemoryDataSet",
}


@lru_cache
def build_extension_map() -> dict[str, list[str]]:
    ext_map = {}
    for typ_ in filetypes:
        cls = get_dataset_cls(typ_)
        for ext in cls.get_supported_extensions():
            try:
                ext_map[ext].append(typ_)
            except KeyError:
                ext_map[ext] = [typ_]
    return ext_map


@typing.overload
def _auto_load(
    path: str, enable_async: Literal[True], *args, executor, **kwargs,
) -> typing.Awaitable[DataSet]:
    ...


@typing.overload
def _auto_load(
    path: str, enable_async: Literal[False], *args, executor, **kwargs,
) -> DataSet:
    ...


@typing.overload
def _auto_load(
    path: str, enable_async: bool, *args, executor, **kwargs,
) -> typing.Union[DataSet, typing.Awaitable[DataSet]]:
    ...


def _auto_load(path, *args, executor, **kwargs):
    if path is None:
        raise DataSetException(
            "please specify the `path` argument to allow auto detection"
        )
    detected_params = detect(path, executor=executor)
    filetype_detected: typing.Optional[str] = detected_params.get('type', None)
    if filetype_detected is None:
        raise DataSetException(
            "could not determine DataSet type for file '%s'" % path,
        )
    return load(
        filetype_detected, path, *args, executor=executor, **kwargs
    )


@typing.overload
def load(
    filetype: str, *args, enable_async: Literal[True], executor, **kwargs,
) -> typing.Awaitable[DataSet]:
    ...


@typing.overload
def load(
    filetype: str, *args, enable_async: Literal[False], executor, **kwargs,
) -> DataSet:
    ...


@typing.overload
def load(
    filetype: str, *args, enable_async: bool, executor, **kwargs,
) -> typing.Union[DataSet, typing.Awaitable[DataSet]]:
    ...


def load(
    filetype: str,
    *args,
    enable_async: bool = False,
    executor,
    **kwargs,
):
    """
    Low-level method to load a dataset. Usually you will want
    to use Context.load instead!

    Parameters
    ----------
    filetype : str or DataSet type
        see libertem.io.dataset.filetypes for supported types, example: 'hdf5'

    executor : JobExecutor

    enable_async : bool
        If True, return a coroutine instead of blocking until the loading has
        finished.

    additional parameters are passed to the concrete DataSet implementation
    """
    if filetype == "auto":
        return _auto_load(*args, executor=executor, enable_async=enable_async, **kwargs)

    cls = get_dataset_cls(filetype)

    async def _init_async():
        ds = cls(*args, **kwargs)
        ds = await sync_to_async(ds.initialize, executor=executor.ensure_sync())
        workers = await executor.get_available_workers()
        scheduler = Scheduler(workers)
        # FIXME the partitioning should be dynamic
        # since the number of eligible workers may depend on
        # the set of UDFs that may or may not run on CPU or GPU
        # This is a workaround with a "best guess compromise"
        ds.set_num_cores(scheduler.effective_worker_count())
        await executor.run_function(ds.check_valid)
        return ds

    if enable_async:
        return _init_async()
    else:
        ds = cls(*args, **kwargs)
        ds = ds.initialize(executor)
        workers = executor.get_available_workers()
        scheduler = Scheduler(workers)
        ds.set_num_cores(scheduler.effective_worker_count())
        executor.run_function(ds.check_valid)
        return ds


def register_dataset_cls(filetype: str, cls: str) -> None:
    filetypes[filetype] = cls


def unregister_dataset_cls(filetype: str) -> None:
    del filetypes[filetype]


def get_dataset_cls(filetype: str) -> type[DataSet]:
    if not isinstance(filetype, str):
        return filetype
    try:
        ft = filetypes[filetype.lower()]
    except KeyError:
        raise DataSetException("unknown filetype: %s" % filetype)
    if not isinstance(ft, str):
        return ft
    parts = ft.split(".")
    module_name = ".".join(parts[:-1])
    cls_name = parts[-1]
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise DataSetException("could not load dataset: %s" % str(e))
    cls: type[DataSet] = getattr(module, cls_name)
    return cls


def get_search_order(path: Union[str, np.ndarray]) -> list[str]:
    """
    Return the keys from filetypes in an order which
    is perhaps optimal for dataset auto-detection
    """
    extension_map = build_extension_map()
    search_order = list(filetypes.keys())
    try:
        # If the file format is registered, float the associated
        # datasets to the top of the search order (maintaining
        # the order in which they were first registered)
        file_format = pathlib.Path(path).suffix.strip().lstrip('.').lower()
        if file_format in extension_map:
            for ds_key in reversed(extension_map[file_format]):
                search_order.pop(search_order.index(ds_key))
                search_order = [ds_key] + search_order
    except (TypeError, ValueError):
        # Let downstream code handle the fact that
        # path cannot be cast to pathlib.Path or provide a suffix
        pass
    try:
        # If path has a shape attribute there is good chance
        # it implements the array interface and as such we should
        # check MemoryDataSet first
        _ = path.shape
        search_order.pop(search_order.index('memory'))
        search_order = ['memory'] + search_order
        warnings.warn('Auto-loading a MemoryDataSet is currently unsupported, '
                      'use ctx.load("memory", data=array).')
    except AttributeError:
        # Cannot interpret as a memory dataset
        pass
    return search_order


def detect(path: Union[str, np.ndarray], executor) -> dict[str, Any]:
    """
    Returns dataset's detected type, parameters and
    additional info.
    """
    search_order = get_search_order(path)
    for filetype in search_order:
        try:
            cls = get_dataset_cls(filetype)
            params = cls.detect_params(path, executor)
        except (NotImplementedError, DataSetException):
            continue
        if not params:
            continue
        params.update({"type": filetype})
        return params
    return {}


def get_extensions() -> set[str]:
    """
    Return supported extensions as a set of strings.

    Plain extensions only, no pattern!
    """
    types: set[str] = set()
    for filetype in filetypes.keys():
        cls = get_dataset_cls(filetype)
        types = types.union({ext.lower() for ext in cls.get_supported_extensions()})
    return types
