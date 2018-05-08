import os
import sys
import json
import h5py
import hdfs3


class Ingestor(object):
    def __init__(self, namenode='localhost', namenode_port=8020, dest_dtype=None):
        self.namenode = namenode
        self.namenode_port = namenode_port
        self.hdfs = hdfs3.HDFileSystem(namenode, port=namenode_port)
        self.dest_dtype = dest_dtype

    def prepare_output(self, output_path_hdfs):
        self.hdfs.mkdir(output_path_hdfs)

    def write_partition(self, data, filename, dtype):
        data = data.astype(dtype)
        fd = self.hdfs.open(filename, "wb", block_size=data.nbytes)
        try:
            bytes_written = fd.write(data.tobytes())
            assert bytes_written == data.nbytes
        finally:
            fd.close()

    def write_index(self, idx, output_filename):
        """
        write json-serializable ``idx`` to ``output_filename`` on hdfs
        """
        fd = self.hdfs.open(output_filename, "wb")
        try:
            idx_bytes = json.dumps(idx).encode("utf8")
            fd.write(idx_bytes)
        finally:
            fd.close()

    def make_partitions_linear(self, reshaped_data, min_num_partitions, target_size):
        first_frame = reshaped_data[0]
        num_frames = reshaped_data.shape[0]
        bytes_per_frame = first_frame.nbytes
        frames_per_partition = target_size // bytes_per_frame
        num_partitions = num_frames // frames_per_partition
        num_partitions = max(min_num_partitions, num_partitions)

        # num_partitions may have changed above, so recalculate
        frames_per_partition = num_frames // num_partitions
        partitions = [
            {"start": i * frames_per_partition,
             "end": (i + 1) * frames_per_partition}
            for i in range(num_partitions)
        ]
        assert partitions[-1]["end"] == num_frames
        return partitions

    def make_index(self, reshaped_data, orig_shape, dtype,
                   min_num_partitions=16, target_size=512*1024*1024):
        """
        create the json-serializable index structure. decides about the
        concrete partitioning, which will later be used to split the input data
        """
        partitions = self.make_partitions_linear(
            reshaped_data=reshaped_data,
            min_num_partitions=min_num_partitions,
            target_size=target_size
        )
        fname_fmt = "partition-%(idx)08d.raw"
        index = {
            "dtype": str(dtype),
            "mode": "linear",  # at the beginning, we only support linear
            "orig_shape": orig_shape,
            "partitions": [
                {
                    "start": p['start'],
                    "end": p['end'],
                    "filename": fname_fmt % {"idx": i},
                }
                for (i, p) in enumerate(partitions)
            ]
        }
        return index

    def write_partitions(self, idx, dataset, output_path_hdfs, dtype):
        for p in idx["partitions"]:
            self.write_partition(
                data=dataset[p["start"]:p["end"]],
                filename=os.path.join(output_path_hdfs, p["filename"]),
                dtype=dtype,
            )

    def main(self, input_filename, input_dataset_path, output_path_hdfs):
        index_fname = os.path.join(output_path_hdfs, "index.json")
        with h5py.File(input_filename, mode="r") as input_f:
            in_ds = input_f[input_dataset_path]
            self.prepare_output(output_path_hdfs)
            s = in_ds.shape
            assert len(s) == 4
            reshaped_data = in_ds.value.reshape(s[0] * s[1], s[2], s[3])
            dest_dtype = self.dest_dtype or in_ds.dtype
            idx = self.make_index(reshaped_data, s, dtype=dest_dtype)
            self.write_index(idx, index_fname)
            print(idx)
            self.write_partitions(
                idx=idx,
                output_path_hdfs=output_path_hdfs,
                dataset=reshaped_data,
                dtype=dest_dtype
            )


if __name__ == "__main__":
    i = Ingestor(dest_dtype='float64')
    i.main(
        input_filename=sys.argv[1],
        input_dataset_path=sys.argv[2],
        output_path_hdfs=sys.argv[3]
    )
