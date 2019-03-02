#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

#define MASK_BIT_PER_ITEM (sizeof(uint32_t) * 8)

inline uint32_t extend_and_mask(uint32_t mask, int bitpos, uint32_t value) {
    uint32_t is_set = (mask & (1 << bitpos)) >> bitpos;      // 1 or 0
    // is_set == 0 -> -is_set = 0
    // is_set == 1 -> -is_set = 0xFFFFFFFF
    return (-is_set) & value;
}

#define A( i, j )     A[ (j)*m + ((i) / MASK_BIT_PER_ITEM) ]
#define B( i, j )     B[ (j)*k + (i) ]
#define C( i, j )     C[ (j)*m + (i) ]


void apply_mask(int m, int n, int k, uint32_t* A, uint32_t* B, uint64_t* C) {
    int i, j, p;
    
    for ( j = 0; j < n; j++ ) {
        for ( p = 0; p < k; p++ ) {
            for ( i = 0; i < m; i++ ) {
                //C(i, j) += A(i, p) * B(p, j);
                C(i + 0, j) += extend_and_mask(A(i + 0, p), (i + 0) % MASK_BIT_PER_ITEM, B(p, j));
                C(i + 1, j) += extend_and_mask(A(i + 1, p), (i + 1) % MASK_BIT_PER_ITEM, B(p, j));
            }
        }
    }
}
