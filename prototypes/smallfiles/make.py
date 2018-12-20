import click
import numpy as np


@click.argument('input_file')
@click.command()
def main(input_file, type=click.File('rb')):

    shape_in = (256, 256, 130, 128)

    # crop the two bottom columns as they contain junk data
    data = np.memmap(input_file, dtype=np.float32).reshape(shape_in)[:, :, :128, :128]

    stack_of_frames = data.reshape((256 * 256, 128, 128))

    for idx, frame in enumerate(stack_of_frames):
        dest = "frame%08d.bin" % idx
        frame.tofile(dest)


if __name__ == "__main__":
    main()
