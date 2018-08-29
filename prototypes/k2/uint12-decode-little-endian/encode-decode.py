{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import math"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3 8\n"
     ]
    }
   ],
   "source": [
    "bits = 12\n",
    "bytebits = 8\n",
    "decoded_types = [np.uint8, np.uint16, np.uint32, np.uint64]\n",
    "decoded_type = None\n",
    "for t in decoded_types:\n",
    "    if np.dtype(t).itemsize*bytebits >= bits:\n",
    "        decoded_type = t\n",
    "        break;\n",
    "decoded_size = np.dtype(decoded_type).itemsize*bytebits\n",
    "\n",
    "if decoded_size <= 32:\n",
    "    encoded_type = np.uint32\n",
    "elif decoded_size <= 64:\n",
    "    encoded_type = np.uint64\n",
    "else:\n",
    "    raise Exception(decoded_size + \" is too many bits\")\n",
    "\n",
    "encoded_size = np.dtype(encoded_type).itemsize*bytebits\n",
    "\n",
    "gcd = math.gcd(bits, encoded_size)\n",
    "\n",
    "# how many words of encoded_type an encoded block\n",
    "encoded_stride = bits // gcd\n",
    "# how many words of \"bits\" bits in an encoded block\n",
    "decoded_stride = encoded_size // gcd\n",
    "\n",
    "print(encoded_stride, decoded_stride)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 1 1 1 1 1 1 1]\n",
      "['0b1', '0b10000', '0b0', '0b1', '0b10000', '0b0', '0b1', '0b10000', '0b0', '0b1', '0b10000', '0b0']\n",
      "[1 1 1 1 1 1 1 1]\n"
     ]
    }
   ],
   "source": [
    "mask = encoded_type(0xfff)\n",
    "\n",
    "decoded = np.zeros(decoded_stride, dtype=decoded_type)\n",
    "decoded_result = np.zeros(decoded_stride, dtype=decoded_type)\n",
    "\n",
    "decoded.fill(1)\n",
    "print(decoded)\n",
    "\n",
    "encoded = np.zeros(encoded_stride, dtype=encoded_type)\n",
    "\n",
    "encoded[0] |= decoded[0] & mask\n",
    "encoded[0] |= (decoded[1] & mask) << 12\n",
    "encoded[0] |= (decoded[2] & mask) << 24\n",
    "\n",
    "encoded[1] |= (decoded[2] & mask) >> 8\n",
    "encoded[1] |= (decoded[3] & mask) << 4\n",
    "encoded[1] |= (decoded[4] & mask) << 16\n",
    "encoded[1] |= (decoded[5] & mask) << 28\n",
    "\n",
    "encoded[2] |= (decoded[5] & mask) >> 4\n",
    "encoded[2] |= (decoded[6] & mask) << 8\n",
    "encoded[2] |= (decoded[7] & mask) << 20\n",
    "\n",
    "print(list(map(bin, encoded.view(np.uint8))))\n",
    "\n",
    "decoded_result[0]  = encoded[0] & mask\n",
    "decoded_result[1]  = (encoded[0] >> 12) & mask\n",
    "decoded_result[2]  = (encoded[0] >> 24) & mask\n",
    "\n",
    "decoded_result[2] |= (encoded[1] <<  8) & mask\n",
    "\n",
    "decoded_result[3]  = (encoded[1] >>  4) & mask\n",
    "decoded_result[4]  = (encoded[1] >> 16) & mask\n",
    "decoded_result[5]  = (encoded[1] >> 28) & mask\n",
    "\n",
    "decoded_result[5] |= (encoded[2] <<  4) & mask\n",
    "\n",
    "decoded_result[6]  = (encoded[2] >>  8) & mask\n",
    "decoded_result[7]  = (encoded[2] >> 20) & mask\n",
    "\n",
    "print(decoded_result)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0 0 0 0xfff\n",
      "1 0 12 0xfff\n",
      "2 0 24 0xff\n",
      "3 1 4 0xfff\n",
      "2 1 -8 0xf00\n",
      "4 1 16 0xfff\n",
      "5 1 28 0xf\n",
      "5 2 -4 0xff0\n",
      "6 2 8 0xfff\n",
      "7 2 20 0xfff\n"
     ]
    }
   ],
   "source": [
    "total_bits = encoded_size * encoded_stride\n",
    "encoded_bytes = np.dtype(decoded_type).itemsize\n",
    "\n",
    "raw_ops = dict()\n",
    "\n",
    "def shift_little(bitindex, bytesize, wordsize):\n",
    "    byte = bitindex // bytesize\n",
    "    rest = bitindex % bytesize\n",
    "    return byte*bytesize - rest + bytesize - 1\n",
    "\n",
    "for b in range(total_bits):\n",
    "    \n",
    "    decoded_index = shift_little(b, bytebits, encoded_bytes) // bits\n",
    "    encoded_index = b // encoded_size\n",
    "    decoded_shift = shift_little(b, bytebits, encoded_bytes) % bits    \n",
    "    encoded_shift = shift_little(b % encoded_size, bytebits, encoded_bytes)\n",
    "    mask = decoded_type(1 << decoded_shift)\n",
    "    net_shift = encoded_shift - decoded_shift\n",
    "    key = (decoded_index, encoded_index, net_shift)\n",
    "    if key in raw_ops:\n",
    "        raw_ops[key] |= mask\n",
    "    else:\n",
    "        raw_ops[key] = mask\n",
    "\n",
    "for key in raw_ops:\n",
    "    (input_index, output_index, net_shift) = key\n",
    "    mask = raw_ops[key]\n",
    "    print(input_index, output_index, net_shift, hex(mask))      \n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
