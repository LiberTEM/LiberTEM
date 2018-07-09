import importlib

filetypes = {
    "hdfs": "libertem.dataset.hdfs.BinaryHDFSDataSet",
    "hdf5": "libertem.dataset.hdf5.H5DataSet",
    "raw": "libertem.dataset.raw.RawFileDataSet",
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
