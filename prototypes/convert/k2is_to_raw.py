import itertools
import logging

import numpy as np
import click

from libertem.io.dataset.k2is import K2ISDataSet


@click.command()
@click.argument('input_filename', type=click.Path())
@click.argument('output_filename', type=click.Path())
@click.argument('scan_size', type=str)
@click.option('--dtype', default='uint16', type=str)
def main(input_filename, output_filename, scan_size, dtype):
    scan_size = tuple(int(x) for x in scan_size.split(","))
    ds = K2ISDataSet(path=input_filename, scan_size=scan_size)

    tile_iters = [p.get_tiles()
                  for p in ds.get_partitions()]

    frame_size = (2 * 930, 8 * 256)

    out_ds = np.memmap(output_filename, dtype=dtype, mode="w+",
                       shape=(scan_size[0] * scan_size[1],) + frame_size)

    for frame in range(scan_size[0] * scan_size[1]):
        for tile_iter in tile_iters:
            for tile in itertools.islice(tile_iter, 32):
                out_ds[frame][tile.tile_slice.get()[2:]] = tile.data.astype(dtype)
        print("frame %d" % frame)
    print("done")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
