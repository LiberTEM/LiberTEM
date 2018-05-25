import click
from .hdf5 import H5Ingestor


@click.group()
def main():
    pass


@main.command()
@click.argument('input_filename', type=click.Path())
@click.argument('input_dataset_path')
@click.argument('output_path_hdfs')
@click.option('--namenode-host', help='hostname of your HDFS namenode', default='localhost')
@click.option('--namenode-port', help='port of your HDFS namenode', default=8020, type=int)
@click.option('--replication', help='number of HDFS replicas', default=3, type=int)
@click.option('--dest-dtype', help='destination datatype, if your data should be converted',
              default=None)
@click.option('--target-partition-size',
              help='partition size in MB; large enough for low overhead, '
                   'small enough for fast feedback',
              default=512)
def hdf5(input_filename, input_dataset_path, output_path_hdfs,
         namenode_host, namenode_port, replication, dest_dtype,
         target_partition_size):
    """
    Ingest the dataset at INPUT_DATASET_PATH in INPUT_FILENAME and save
    it to HDFS at OUTPUT_PATH_HDFS (will be created as a directory)
    """
    i = H5Ingestor(
        namenode=namenode_host,
        namenode_port=namenode_port,
        replication=replication,
    )
    i.main(
        input_filename=input_filename,
        input_dataset_path=input_dataset_path,
        output_path_hdfs=output_path_hdfs,
        dest_dtype=dest_dtype,
        target_partition_size=target_partition_size*1024*1024,
    )
