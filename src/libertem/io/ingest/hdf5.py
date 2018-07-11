import os

import h5py

from .sink import HDFSBinarySink


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
        self.sink = HDFSBinarySink(
            namenode=namenode,
            namenode_port=namenode_port,
            replication=replication
        )

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
            self.sink.prepare_output(output_path_hdfs)
            s = in_ds.shape
            assert len(s) == 4
            dest_dtype = dest_dtype or in_ds.dtype
            idx = self.sink.make_index(
                data=in_ds,
                dtype=dest_dtype,
                target_size=target_partition_size,
            )
            self.sink.write_index(idx, index_fname)
            self.sink.write_partitions(
                idx=idx,
                output_path_hdfs=output_path_hdfs,
                dataset=in_ds,
                dtype=dest_dtype
            )
