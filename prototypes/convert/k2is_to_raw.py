import logging

import numpy as np
import click

from libertem.io.dataset.k2is import K2ISDataSet


@click.command()
@click.argument('input_filename', type=click.Path())
@click.argument('output_filename', type=click.Path())
@click.option('--dtype', default='uint16', type=str)
def main(input_filename, output_filename, dtype):
    ds = K2ISDataSet(path=input_filename)
    ds.initialize()

    out_ds = np.memmap(output_filename, dtype=dtype, mode="w+",
                       shape=tuple(ds.raw_shape))

    num_parts = len(list(ds.get_partitions()))

    for p_idx, p in enumerate(ds.get_partitions()):
        for tile in p.get_tiles():
            out_ds[tile.tile_slice.get()] = tile.data.astype(dtype)
        print("partition %d/%d done" % (p_idx + 1, num_parts))
    print("done")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
