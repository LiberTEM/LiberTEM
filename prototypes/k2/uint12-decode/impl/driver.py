from __future__ import division

import numpy as np
import u12mod
import mmap


def main():
    filename = "/home/alex/Data/Capture52_1_slice.bin"
    f = open(filename, 'rb')

    input_data = mmap.mmap(
        f.fileno(),
        length=0x5758,  # one block (incl. header)
        access=mmap.ACCESS_READ,
        offset=0,
    )

    out2 = np.zeros(930*16, dtype="uint16")

    u12mod.decode_uint12_cpp_uint16_naive(inp=input_data[40:], out=out2)

    print("out2=", out2)


main()
