import os

import numpy as np

from .sink import HDFSBinarySink

# FIXME: extract common code from ingestors into an I/O module


class EMPADIngestor(object):
    def __init__(self, namenode='localhost', namenode_port=8020, replication=3):
        """
        Move data from a local EMPAD raw file to a HDFS filesystem

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

    def main(self, input_filename, output_path_hdfs, target_partition_size,
             shape_in=(256, 256, 130, 128), crop_to=(128, 128),
             src_dtype="float32", dest_dtype=None):
        """
        Ingest data from ``input_filename`` EMPAD raw file to HFDS

        Parameters
        ----------
        input_filename : str
            path to EMPAD raw input file
        output_path_hdfs : str
            output HDFS path (will be created as a directory)
        src_dtype : str (default: float32)
            input datatype
        dest_dtype : str or None
            convert to this datatype while ingesting (default: keep input datatype)
        target_partition_size : int
            target partition size in bytes
        shape_in : (int, int, int, int)
            shape of the raw input
        crop_to : (int, int)
            crop frames to this sensor size
        """
        index_fname = os.path.join(output_path_hdfs, "index.json")
        with open(input_filename, mode="r") as input_f:
            in_ds = np.fromfile(input_f, dtype=src_dtype)
            in_ds = in_ds.reshape(shape_in)[:, :, :crop_to[0], :crop_to[1]]
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
