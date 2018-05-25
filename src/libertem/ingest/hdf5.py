import os
import json

import h5py
import hdfs3

from libertem.slice import Slice


class H5Ingestor(object):
    def __init__(self, namenode='localhost', namenode_port=8020, replication=3):
        """
        Move data from a local HDF5 file to a HDFS filesystem

        Parameters
        ----------
        namenode : str
            hostname of the HDFS namenode
        namenode_port : int
            port of the HDFS namenode (default 8020)
        replication : int
            number of replicas (default 3)
        """
        self.namenode = namenode
        self.namenode_port = namenode_port
        self.hdfs = hdfs3.HDFileSystem(namenode, port=namenode_port)
        self.replication = replication

    def main(self, input_filename, input_dataset_path, output_path_hdfs,
             target_partition_size, dest_dtype=None):
        """
        Ingest data from ``input_filename`` HDF5 file to HFDS

        Parameters
        ----------
        input_filename : str
            path to HDF5 input file
        input_dataset_path : str
            path to the dataset inside the HDF5 file you want to ingest
        output_path_hdfs : str
            output HDFS path (will be created as a directory)
        dest_dtype : str or None
            convert to this datatype while ingesting (default: keep input datatype)
        target_partition_size : int
            target partition size in bytes
        """
        index_fname = os.path.join(output_path_hdfs, "index.json")
        with h5py.File(input_filename, mode="r") as input_f:
            in_ds = input_f[input_dataset_path]
            self.prepare_output(output_path_hdfs)
            s = in_ds.shape
            assert len(s) == 4
            dest_dtype = dest_dtype or in_ds.dtype
            idx = self.make_index(
                data=in_ds,
                dtype=dest_dtype,
                target_size=target_partition_size,
            )
            self.write_index(idx, index_fname)
            self.write_partitions(
                idx=idx,
                output_path_hdfs=output_path_hdfs,
                dataset=in_ds,
                dtype=dest_dtype
            )

    def prepare_output(self, output_path_hdfs):
        # FIXME: check for existance of output directory, bail out if exists
        # FIXME: add force=True param to force even if output path exists
        self.hdfs.mkdir(output_path_hdfs)

    def write_partition(self, data, filename, dtype):
        data = data.astype(dtype)
        fd = self.hdfs.open(filename, "wb", block_size=data.nbytes, replication=self.replication)
        try:
            assert data.tobytes() is not None
            bytes_written = fd.write(data.tobytes())
            assert bytes_written is not None
            assert bytes_written == data.nbytes, "%d != %d" % (bytes_written, data.nbytes)
        finally:
            fd.close()

    def write_index(self, idx, output_filename):
        """
        write json-serializable ``idx`` to ``output_filename`` on hdfs
        """
        fd = self.hdfs.open(output_filename, "wb", replication=self.replication)
        try:
            idx_bytes = json.dumps(idx).encode("utf8")
            fd.write(idx_bytes)
        finally:
            fd.close()

    def make_partitions(self, data, partition_shape):
        assert data.shape[0] % partition_shape[0] == 0
        assert data.shape[1] % partition_shape[1] == 0
        partitions = [
            {"origin": (y * partition_shape[0], x * partition_shape[1]),
             "shape": partition_shape}
            for x in range(data.shape[1] // partition_shape[1])
            for y in range(data.shape[0] // partition_shape[0])
        ]
        return partitions

    def make_index(self, data, dtype, min_num_partitions=16, target_size=512*1024*1024):
        """
        create the json-serializable index structure. decides about the
        concrete partitioning, which will later be used to split the input data
        """
        partition_shape = Slice.partition_shape(
            datashape=data.shape,
            framesize=data[0][0].size,
            dtype=dtype,
            min_num_partitions=min_num_partitions,
            target_size=target_size,
        )
        partitions = self.make_partitions(
            data=data,
            partition_shape=partition_shape,
        )
        fname_fmt = "partition-%(idx)08d.raw"
        index = {
            "dtype": str(dtype),
            "mode": "rect",
            "shape": data.shape,
            "partitions": [
                {
                    "origin": p['origin'],
                    "shape": p['shape'],
                    "filename": fname_fmt % {"idx": i},
                }
                for (i, p) in enumerate(partitions)
            ]
        }
        return index

    def write_partitions(self, idx, dataset, output_path_hdfs, dtype):
        for p in idx["partitions"]:
            dataslice = dataset[
                p['origin'][0]:(p['origin'][0] + p['shape'][0]),
                p['origin'][1]:(p['origin'][1] + p['shape'][1]), :, :]
            assert dataslice.shape == p['shape'],\
                "%r != %r (origin=%r)" % (dataslice.shape, p['shape'], p['origin'])
            self.write_partition(
                data=dataslice,
                filename=os.path.join(output_path_hdfs, p["filename"]),
                dtype=dtype,
            )
