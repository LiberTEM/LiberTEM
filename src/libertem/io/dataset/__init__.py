import importlib

filetypes = {
    "hdfs": "libertem.io.dataset.hdfs.BinaryHDFSDataSet",
    "hdf5": "libertem.io.dataset.hdf5.H5DataSet",
    "raw": "libertem.io.dataset.raw.RawFileDataSet",
    "mib": "libertem.io.dataset.mib.MIBDataSet",
}


def load(filetype, *args, **kwargs):
    try:
        ft = filetypes[filetype.lower()]
    except KeyError:
        raise ValueError("unknown filetype: %s" % filetype)
    parts = ft.split(".")
    module = ".".join(parts[:-1])
    cls = parts[-1]
    module = importlib.import_module(module)
    cls = getattr(module, cls)
    return cls(*args, **kwargs)
