import click
from libertem.cli_tweaks import console_tweaks
from .hdf5 import H5Ingestor
from .empad import EMPADIngestor


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
    console_tweaks()
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


@main.command()
@click.argument('input_filename', type=click.Path())
@click.argument('output_path_hdfs')
@click.option('--namenode-host', help='hostname of your HDFS namenode', default='localhost')
@click.option('--namenode-port', help='port of your HDFS namenode', default=8020, type=int)
@click.option('--replication', help='number of HDFS replicas', default=3, type=int)
@click.option('--shape-in', help='shape of raw input data', default='256,256,130,128', type=str)
@click.option('--src-dtype', help='dtype of raw input data', default='float32', type=str)
@click.option('--crop-to', help='crop frames to this size', default='128,128', type=str)
@click.option('--dest-dtype', help='destination datatype, if your data should be converted',
              default=None)
@click.option('--target-partition-size',
              help='partition size in MB; large enough for low overhead, '
                   'small enough for fast feedback',
              default=512)
def empad(input_filename, output_path_hdfs, shape_in, src_dtype, crop_to,
         namenode_host, namenode_port, replication, dest_dtype,
         target_partition_size):
    """
    Ingest the dataset in INPUT_FILENAME and save
    it to HDFS at OUTPUT_PATH_HDFS (will be created as a directory)
    """
    # TODO: maybe read the XML file to find scan dimensions etc.?
    console_tweaks()
    i = EMPADIngestor(
        namenode=namenode_host,
        namenode_port=namenode_port,
        replication=replication,
    )
    i.main(
        input_filename=input_filename,
        output_path_hdfs=output_path_hdfs,
        dest_dtype=dest_dtype,
        target_partition_size=target_partition_size*1024*1024,
        shape_in=tuple(int(p) for p in shape_in.split(",")),
        src_dtype=src_dtype,
        crop_to=tuple(int(p) for p in crop_to.split(",")),
    )
