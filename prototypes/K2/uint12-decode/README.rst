DECODING 12 BIT UINTs
=====================

Image data of the Gatan K2 camera is encoded as 12 bit packed unsigned integers, i.e. each
pixel value occupies 1.5 bytes. Since this format cannot be processed directly by the CPU, it has 
to be unpacked to a suitable format first, for example as uint16.

On CPU-native raw data, the LiberTEM processing back-end achieves staggering throughput 
beyond 12 GB/s with suitable numerics packages like a BLAS implementation or pytorch. An 
optimized 12 bit decoder is one critical component to achieve such performance levels with 
the native K2 format.

The simple algorithm in unpack-12-alex.py achieves decent performance of about 1 GB/s with numba.
However, this is an order of magnitude slower than the speed at which LiberTEM processes suitable
data directly. For that reason we have investigated options to increase speed.

Why the simple algorithm is limited
-----------------------------------

Modern CPUs achieve their optimal throughput when they can use their SIMD instruction sets, 
optimize cache use and read data in predictable patterns aligned with 32-bit or 64-bit boundaries, 
among other things. The simple algorithm is problematic in that context: It reads the input in 
three-byte portions so that each loop iteration reads with a different alignment relative to 32-bit or 
64-bit boundaries. Furthermore, the number of iterations is unknown at compile time. Such a pattern 
makes it hard for a compiler to vectorize the loop, i.e. combine blocks of several loop iterations 
into sets of SIMD instructions.

Approach to vectorize bit unpacking
-----------------------------------

The bit pattern of the 12 bit data stream aligns with the 32-bit or 64-bit boundaries in a repeating
fashion. Specifically, every 96/192 bits (3x uint32/uint64; 8/16 x 12 bit), the input stream and the 
CPU-native view on the data have the same alignment. The general case for any bit-length of the input
data can be calculated by dividing the source resp. working bit length by their greatest common divisor
(gcd). This yields the number of data words in a block, seen from working resp. source perspective.

That means each such 96/192 blocks can be processed with the same sequence of CPU instructions, 
i.e. SIMD instructions can be used to process several blocks together in parallel. 
In theory, an advanced compiler could perhaps in the future recognize such a pattern without human 
intervention. In practice, the code has to be written explicitly to process the data in such 
blocks so that the compiler can recognize and vectorize the pattern.

The conversion gets more complicated with the endian-ness of input data and CPU interpretation. Since x86
CPUs are little-endian, i.e. have the lowest order byte at the lowest address in a data word, the bit significance
in a 32 bit uint is 7 6 5 4 3 2 0 | 15 14 13 12 11 10 9 8 | 23 22 21 20 19 18 17 16 | 31 30 29 28 27 26 25 24, while
the input pattern follows 11 10 9 8 7 6 5 4 | 3 2 1 0 - 11 10 9 8 | 7 6 5 4 3 2 1 0 and so on. 
If we want good performance and use SIMD instructions, the idea is to read the input data aligned as 
uint32 or uint64. Unfortunately, that means the various bits of the input stream have to 
be fished out in smaller units using AND with a mask, shifted, and finally ORd their appropriate 
place in the uint16 output word. That is very much possible with SIMD instructions, but 
hard to understand for a human. 

The Python code pack-12.py and various C/C++ versions contain code that calculate the appropriate indices,
shifts and masks to map every bit of an input block to their appropriate place in the output sequence. 
pack-12.py, unpack-12-7.cpp and unpack-12-4.cpp contain code that can cover the general case,
i.e. arbitrary combinations of input bit length and working registers. This code is indeed 
vectorized by gcc. However, this is likely not optimal yet. In many cases several single-bit 
operations can be ORd together when they have same indices and shifts, and only differ by 
their mask bit. The Python code builds an optimized list of operations with combined masks for that purpose.
In the compiled version of unpack-12-7.cpp and unpack-12-4.cpp, there was no indication that the
optimizer recognized that several operations can be grouped -- but that's not known for sure. 
For that reason versions with hand-written code that uses the optimized sequence for 12 bit 
calculated with the Python version were written. It remains to be tested which version is 
actually faster, in the end.

General considerations for optimizer-friendly code
--------------------------------------------------

Any compiler optimization works best when as much as possible is known or precalculated at 
compile time. The C/C++ code contains various measures that should help the compiler with 
that task. The code generally contains intermediate loops with a number of iterations that is 
known at compile time to help with vectorization and test for a potential "sweet spot" for the block size.
Code to cover an unaligned remaining block is omitted for simplicity. Helper functions are 
designed and declared as constexpr to motivate the compiler to the maximum to evaluate 
them at compile time. That is indeed successful: much of the source code is not part of the 
output machine code or is inlined as compact instruction sequences if the function input 
values depend on the input data. Constants are used instead of variables where possible.

It seemed to help the optimizer when input and output data are not read / written directly 
from/to the buffer, but when they are first copied to a local array, processed into a second 
local array and then copied to the output array.

The various implementations in C++
----------------------------------

How to group and order the operations for a block has room for variation.
The inner loop can process an entire block, or several inner loops are executed sequentially 
to process a specific word of the input or output data in each block for a number of 
blocks in a row. These possibilities are explored in the various C++ implementations with the goal to see
what the compiler makes of them. Furthermore, data types and block sizes can be varied
in the code by changing the appropriate values.

None of the implementations are tested for correctness. They likely contain issues with 
index calculation that don't show up on compile time. Furthermore, they are not 
benchmarked yet. The main goal was to explore how to write code that is vectorized and optimized.

Exploring the compiled result
-----------------------------

Have a look at <https://godbolt.org/>. Copy and paste the code, select language and compiler,
and see what machine code is generated.

Conclusion
----------

The modified algorithm can indeed be vectorized by the compiler in several of the variations.
Which one is faster is far from certain. That might depend on many different factors, including
cache efficiency for code and data. It is even possible that the simple implementation is the 
fastest, in the end. The work presented here can perhaps serve as a starting point to 
investigate this further.
