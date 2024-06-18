import sys

import numpy as np
import numba


@numba.njit(inline='always', cache=True)
def byteswap_2_straight(inp, out):
    for i in range(inp.shape[0] // 2):
        out[i * 2 + 0] = inp[i * 2 + 1]
        out[i * 2 + 1] = inp[i * 2 + 0]


@numba.njit(inline='always', cache=True)
def byteswap_2_decode(inp, out):
    for i in range(inp.shape[0] // 2):
        o0 = np.uint16(inp[i * 2 + 0]) << 8
        o1 = np.uint16(inp[i * 2 + 1]) << 0
        out[i] = o0 | o1


@numba.njit(inline='always', cache=True)
def byteswap_4_straight(inp, out):
    for i in range(inp.shape[0] // 4):
        out[i * 4 + 3] = inp[i * 4 + 0]
        out[i * 4 + 2] = inp[i * 4 + 1]
        out[i * 4 + 1] = inp[i * 4 + 2]
        out[i * 4 + 0] = inp[i * 4 + 3]


@numba.njit(inline='always', cache=True)
def byteswap_4_decode(inp, out):
    for i in range(inp.shape[0] // 4):
        o0 = np.uint32(inp[i * 4 + 0]) << 24
        o1 = np.uint32(inp[i * 4 + 1]) << 16
        o2 = np.uint32(inp[i * 4 + 2]) << 8
        o3 = np.uint32(inp[i * 4 + 3]) << 0
        out[i] = o0 + o1 + o2 + o3


@numba.njit(inline='always', cache=True)
def byteswap_8_straight(inp, out):
    for i in range(inp.shape[0] // 8):
        out[i * 8 + 7] = inp[i * 8 + 0]
        out[i * 8 + 6] = inp[i * 8 + 1]
        out[i * 8 + 5] = inp[i * 8 + 2]
        out[i * 8 + 4] = inp[i * 8 + 3]
        out[i * 8 + 3] = inp[i * 8 + 4]
        out[i * 8 + 2] = inp[i * 8 + 5]
        out[i * 8 + 1] = inp[i * 8 + 6]
        out[i * 8 + 0] = inp[i * 8 + 7]


@numba.njit(inline='always', cache=True)
def byteswap_8_decode(inp, out):
    for i in range(inp.shape[0] // 8):
        o0 = np.uint64(inp[i * 8 + 0]) << 56
        o1 = np.uint64(inp[i * 8 + 1]) << 48
        o2 = np.uint64(inp[i * 8 + 2]) << 40
        o3 = np.uint64(inp[i * 8 + 3]) << 32
        o4 = np.uint64(inp[i * 8 + 4]) << 24
        o5 = np.uint64(inp[i * 8 + 5]) << 16
        o6 = np.uint64(inp[i * 8 + 6]) << 8
        o7 = np.uint64(inp[i * 8 + 7]) << 0
        out[i] = o0 + o1 + o2 + o3 + o4 + o5 + o6 + o7


@numba.njit(inline='always', cache=True)
def default_decode(inp, out, idx, native_dtype, rr, origin, shape, ds_shape):
    out[idx, :] = inp.view(native_dtype)


@numba.njit(inline='always', cache=True)
def decode_swap_2(inp, out, idx, native_dtype, rr, origin, shape, ds_shape):
    byteswap_2_decode(inp, out=out[idx])


@numba.njit(inline='always', cache=True)
def decode_swap_4(inp, out, idx, native_dtype, rr, origin, shape, ds_shape):
    byteswap_4_decode(inp, out=out[idx])


@numba.njit(inline='always', cache=True)
def decode_swap_8(inp, out, idx, native_dtype, rr, origin, shape, ds_shape):
    byteswap_8_decode(inp, out=out[idx])


@numba.njit(inline='always', cache=True)
def decode_swap_only_2(inp, out, idx, native_dtype, rr, origin, shape, ds_shape):
    byteswap_2_straight(inp, out=out[idx].view(np.uint8))


@numba.njit(inline='always', cache=True)
def decode_swap_only_4(inp, out, idx, native_dtype, rr, origin, shape, ds_shape):
    byteswap_4_straight(inp, out=out[idx].view(np.uint8))


@numba.njit(inline='always', cache=True)
def decode_swap_only_8(inp, out, idx, native_dtype, rr, origin, shape, ds_shape):
    byteswap_8_straight(inp, out=out[idx].view(np.uint8))


def _convert_byteorder_eq(order):
    if order != '=':
        return order
    return {
        'little': '<',
        'big': '>',
    }[sys.byteorder]


class Decoder:
    def do_clear(self):
        return False

    def get_native_dtype(self, inp_native_dtype, read_dtype):
        return inp_native_dtype

    def get_decode(self, native_dtype, read_dtype):
        raise NotImplementedError()


class DtypeConversionDecoder(Decoder):
    def _need_byteswap(self, native_dtype, read_dtype):
        native_dtype = np.dtype(native_dtype)
        read_dtype = np.dtype(read_dtype)
        nd_order = _convert_byteorder_eq(native_dtype.byteorder)
        rd_order = _convert_byteorder_eq(read_dtype.byteorder)
        return nd_order != rd_order and native_dtype.itemsize > 1

    def _swapping_decode(self, native_dtype):
        return {
            2: decode_swap_2,
            4: decode_swap_4,
            8: decode_swap_8,
        }[native_dtype.itemsize]

    def _swap_only_decode(self, native_dtype):
        return {
            2: decode_swap_only_2,
            4: decode_swap_only_4,
            8: decode_swap_only_8,
        }[native_dtype.itemsize]

    def get_decode(self, native_dtype, read_dtype):
        native_dtype = np.dtype(native_dtype)
        read_dtype = np.dtype(read_dtype)
        if not self._need_byteswap(native_dtype, read_dtype):
            return default_decode

        if native_dtype.kind in ('f', 'c'):
            # TODO: can implement f32->f32 and f64->f64 by straight swapping, and
            # other conversions via a two-step decoding process
            raise NotImplementedError(
                "byte swapping for floats not implemented yet"
            )

        return self._swapping_decode(native_dtype)

    def get_native_dtype(self, inp_native_dtype, read_dtype):
        if self._need_byteswap(inp_native_dtype, read_dtype):
            return np.dtype(np.uint8)
        return np.dtype(inp_native_dtype)
