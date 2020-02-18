import importlib

from libertem.io.dataset.base import DataSetException


filetypes = {
    "hdf5": "libertem.io.dataset.hdf5.H5DataSet",
    "raw": "libertem.io.dataset.raw.RawFileDataSet",
    "mib": "libertem.io.dataset.mib.MIBDataSet",
    "blo": "libertem.io.dataset.blo.BloDataSet",
    "k2is": "libertem.io.dataset.k2is.K2ISDataSet",
    "ser": "libertem.io.dataset.ser.SERDataSet",
    "frms6": "libertem.io.dataset.frms6.FRMS6DataSet",
    "empad": "libertem.io.dataset.empad.EMPADDataSet",
    "memory": "libertem.io.dataset.memory.MemoryDataSet",
    "dm": "libertem.io.dataset.dm.DMDataSet",
}


def load(filetype, executor, *args, **kwargs):
    """
    Low-level method to load a dataset. Usually you will want
    to use Context.load instead!

    Parameters
    ----------
    filetype : str or DataSet type
        see libertem.io.dataset.filetypes for supported types, example: 'hdf5'

    executor : JobExecutor

    additional parameters are passed to the concrete DataSet implementation
    """
    cls = get_dataset_cls(filetype)
    ds = cls(*args, **kwargs)
    ds = ds.initialize(executor)
    ds.set_num_cores(len(executor.get_available_workers()))
    executor.run_function(ds.check_valid)
    return ds


def register_dataset_cls(filetype, cls):
    filetypes[filetype] = cls


def unregister_dataset_cls(filetype):
    del filetypes[filetype]


def get_dataset_cls(filetype):
    if not isinstance(filetype, str):
        return filetype
    try:
        ft = filetypes[filetype.lower()]
    except KeyError:
        raise DataSetException("unknown filetype: %s" % filetype)
    if not isinstance(ft, str):
        return ft
    parts = ft.split(".")
    module = ".".join(parts[:-1])
    cls = parts[-1]
    try:
        module = importlib.import_module(module)
    except ImportError as e:
        raise DataSetException("could not load dataset: %s" % str(e))
    cls = getattr(module, cls)
    return cls


def detect(path, executor):
    for filetype in filetypes.keys():
        try:
            cls = get_dataset_cls(filetype)
            maybe_params = cls.detect_params(path, executor)
        except (NotImplementedError, DataSetException):
            continue
        if not maybe_params:
            continue
        params = {}
        params.update(maybe_params)
        params.update({"type": filetype})
        return params
    return {}
