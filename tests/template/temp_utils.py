def _get_hdf5_params(path):
    return {
        "type": "HDF5",
        "params": {
            "path": path,
            "ds_path": "/data"
            },
    }
