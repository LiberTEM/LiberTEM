import importlib

filetypes = {
    "hdfs": "libertem.io.dataset.hdfs.BinaryHDFSDataSet",
    "hdf5": "libertem.io.dataset.hdf5.H5DataSet",
    "raw": "libertem.io.dataset.raw.RawFileDataSet",
    "mib": "libertem.io.dataset.mib.MIBDataSet",
    "blo": "libertem.io.dataset.blo.BloDataSet",
    "k2is": "libertem.io.dataset.k2is.K2ISDataSet",
}


def load(filetype, *args, **kwargs):
    """
    load a dataset

    Parameters
    ----------
    filetype : str
        see libertem.io.dataset.filetypes for supported types, example: 'hdf5'

    additional parameters are passed to the concrete DataSet implementation
    """
    cls = _get_dataset_cls(filetype)
    return cls(*args, **kwargs)


def _get_dataset_cls(filetype):
    try:
        ft = filetypes[filetype.lower()]
    except KeyError:
        raise ValueError("unknown filetype: %s" % filetype)
    parts = ft.split(".")
    module = ".".join(parts[:-1])
    cls = parts[-1]
    module = importlib.import_module(module)
    cls = getattr(module, cls)
    return cls


def detect(path):
    for filetype in filetypes.keys():
        cls = _get_dataset_cls(filetype)
        try:
            maybe_params = cls.detect_params(path)
        except NotImplementedError:
            continue
        if not maybe_params:
            continue
        params = {}
        params.update(maybe_params)
        params.update({"type": filetype})
        return params
    return {}
