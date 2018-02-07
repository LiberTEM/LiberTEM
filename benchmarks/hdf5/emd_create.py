import h5py
import numpy as np


def create_emd_file(filename, chunks):
    f = h5py.File(filename, mode="w")
    f.attrs.create('version_major', 0)
    f.attrs.create('version_minor', 2)

    f.create_group('experimental/science_data')
    group = f['experimental/science_data']
    group.attrs.create('emd_group_type', 1)

    data = np.random.random((128, 128, 128, 128))
    group.create_dataset(name='data', data=data, chunks=chunks)
    group.create_dataset(name='dim1', data=range(128))
    group['dim1'].attrs.create('name', b'dim1')
    group['dim1'].attrs.create('units', b'units1')
    group.create_dataset(name='dim2', data=range(128))
    group['dim2'].attrs.create('name', b'dim2')
    group['dim2'].attrs.create('units', b'units2')
    group.create_dataset(name='dim3', data=range(128))
    group['dim3'].attrs.create('name', b'dim3')
    group['dim3'].attrs.create('units', b'units3')
    group.create_dataset(name='dim4', data=range(128))
    group['dim4'].attrs.create('name', b'dim4')
    group['dim4'].attrs.create('units', b'units4')
    f.close()


create_emd_file("test_chunks_01_01_128_128.emd", (1, 1, 128, 128))
create_emd_file("test_chunks_02_02_128_128.emd", (2, 2, 128, 128))
create_emd_file("test_chunks_04_04_128_128.emd", (4, 4, 128, 128))
create_emd_file("test_chunks_12_12_128_128.emd", (12, 12, 128, 128))
create_emd_file("test_chunks_24_24_128_128.emd", (24, 24, 128, 128))
create_emd_file("test_chunks_48_48_128_128.emd", (48, 48, 128, 128))
