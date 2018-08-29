import numpy as np
import math
import time

# Calculate the significance of a bit in a little-endian word, counting from the left
def shift_little(bitindex, bytesize, wordsize):
    byte = bitindex // bytesize
    rest = bitindex % bytesize
    return byte*bytesize - rest + bytesize - 1

# Calculate the significance of a bit in a big-endian word, counting from the left    
def shift_big(bitindex, bytesize, wordsize):
    return wordsize * bytesize - bitindex - 1

# Apply operations in ops in encoding directions    
def encode_12(inp, out, ops):
    for (input_index, output_index, net_shift, mask) in ops:
        out[output_index] |= worktype((inp[input_index] & mask) * 2**net_shift)

# Apply operations in ops in decoding directions    
def decode_12(inp, out, ops):
    for (output_index, input_index, net_shift, mask) in ops:
        out[output_index] |= inputtype(inp[input_index] // 2**net_shift) & mask

bits = 12
bytebits = 8
inputtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
inputtype = None
for t in inputtypes:
    if np.dtype(t).itemsize*bytebits >= bits:
        inputtype = t
        break;
inputsize = np.dtype(inputtype).itemsize*bytebits

if inputsize <= 32:
    worktype = np.uint32
elif inputsize <= 64:
    worktype = np.uint64
else:
    raise Exception(inputsize + " is too many bits")

worksize = np.dtype(worktype).itemsize*bytebits

gcd = math.gcd(bits, worksize)

workstride = bits // gcd
inputstride = worksize // gcd

# print(workstride, inputstride)

total_bits = worksize * workstride


raw_ops = dict()

ops = []

# calculate the decoded index, encoded index, shift and mask for each 
# bit in a block
# OR bits that have the same indices and shift together.
for b in range(total_bits):
    input_index = b // bits
    output_index = b // worksize
    input_shift = shift_big(b % bits, bits, 1)
    output_shift = shift_little(b % worksize, bytebits, worksize)
    mask = inputtype(1 << input_shift)
    net_shift = output_shift - input_shift
    key = (input_index, output_index, net_shift)
    if key in raw_ops:
        raw_ops[key] |= mask
    else:
        raw_ops[key] = mask

# Write it; 
# convert to numba-friendly list instead of dict
for key in raw_ops:
    (input_index, output_index, net_shift) = key
    mask = raw_ops[key]
    print(input_index, output_index, net_shift, hex(mask))
    # ops.append((input_index, output_index, net_shift, mask))

# testinputsize = workstride*worksize*1024*1024

# input = np.arange(testinputsize, dtype=inputtype)
# reverse = np.zeros(testinputsize, dtype=inputtype)

# output = np.zeros((workstride, testinputsize*bits//worksize//workstride), dtype=worktype)

# print(output)

# start = time.time()

# input = input.reshape((testinputsize//inputstride, inputstride)).T
# reverse = reverse.reshape((testinputsize//inputstride, inputstride)).T

# encode_12(input, output, ops)

# print(output)        
    
# output = output.T.flatten()
# print(output)    
# print(list(map(bin, output.view(np.uint8))))

# output = output.reshape((testinputsize*bits//worksize//workstride, workstride)).T

# print(output)
# decode_12(output, reverse, ops)
   
# reverse = reverse.T.flatten()

# print(input.size / (time.time() - start))
 
# Sample for manual algorithm for better understanding
 
testwork = np.zeros(3, dtype=np.uint32)

vals = np.zeros(8, dtype=np.uint16)

vals.fill(1 << 8)
masks = np.zeros(8)
twelvebit = 0xfff
masks[0] = twelvebit
# masks[1] <<= (32 - 24)
# masks[2] >>= (36 - 32)
# masks[3] <<= (32 - 4)
# masks[4] <<= (32 - 16)
# masks[5] <<= (32 - 28)
# masks[6] >>= (40 - 32)
# masks[7] <<= (32 - 8)
# masks[8] <<= (32 - 20)

#              1              1              1 
#             #0             #1             #2             #3             #4             #5             #6             #7
#         a|   b|   a|        b|        c|   d|   c|        d|        e|   f|   e|        f|        g|   h|   g|        h|
# 0000 0001 0001 0000 0000 0000 0000 0001 0001 0000 0000 0000 0000 0001 0001 0000 0000 0000 0000 0001 0001 0000 0000 0000 
#          |         |         |         ||        |         |         |        ||         |         |         |        ||
#         1        16         0         1        16         0         1        16         0         1        16         0

#              #0            #1             #2             #3             #4             #5             #6             #7
#               |              |              |              |              |              |              |              |
# 0000 0001 0000 0000 0001 0000 0000 0001 0000 0000 0001 0000 0000 0001 0000 0000 0001 0000 0000 0001 0000 0000 0001 0000 
#          |         |         |         ||        |         |         |        ||         |         |         |        ||
#         1         0        16         1         0        16         1         0        16         1        0        16

#              #0            #1             #2             #3             #4             #5             #6             #7
#               |              |              |              |              |              |              |              |
# 0001 0000 0000 0001 0000 0000 0001 0000 0000 0001 0000 0000 0001 0000 0000 0001 0000 0000 0001 0000 0000 0001 0000 0000 
#          |         |         |         ||        |         |         |        ||         |         |         |        ||
#        16         1         0         16         1         0        16         1         0        16         1         0

# testwork[0] |= (vals[0] & (quad3 | quad2)) >> 4
# testwork[0] |= (vals[0] & quad1) << 12
# testwork[0] |= (vals[1] & quad3)
# testwork[0] |= (vals[1] & (quad2 | quad1)) << 16
# testwork[0] |= (vals[2] & (quad3 | quad2)) << 20
# testwork[1] |= (vals[2] & quad1) << 4
# testwork[1] |= (vals[3] & quad3) >> 8
# testwork[1] |= (vals[3] & (quad2 | quad1)) << 8
# testwork[1] |= (vals[4] & (quad3 | quad2)) << 12
# testwork[1] |= (vals[4] & quad1) << 28
# testwork[1] |= (vals[5] & quad3) << 16
# testwork[2] |= (vals[5] & (quad2 | quad1))
# testwork[2] |= (vals[6] & (quad3 | quad2)) << 4
# testwork[2] |= (vals[6] & quad1) << 20
# testwork[2] |= (vals[7] & quad3) << 8
# testwork[2] |= (vals[7] & (quad2 | quad1)) << 24

# testwork[0] |= (vals[0]) 
# testwork[0] |= vals[1]
# testwork[0] |= vals[2]
# testwork[1] |= vals[3]
# testwork[1] |= vals[4]
# testwork[1] |= vals[5]
# testwork[1] |= vals[6]
# testwork[2] |= vals[7]


#testwork.byteswap(inplace=True)

# print(list(map(bin, testwork.view(np.uint8))))
        
# print(input)
