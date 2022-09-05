import numpy as np
import cupy as cp
import cupyx

for dtype in np.complex64, np.complex128:
    for sparse_cls in (
        cupyx.scipy.sparse.csr_matrix, cupyx.scipy.sparse.csc_matrix, cupyx.scipy.sparse.coo_matrix
    ):
        a = cp.array([
            (1, 1j),
            (1j, -1-1j)

        ]).astype(dtype)
        sp = sparse_cls(a)
        check = sp.todense()
        if not np.allclose(a, check):
            print(dtype, sparse_cls, "*** RESULTS DIFFER ***")
            print("nonzeros", a[a != 0])
            print("data", sp.data)
            print("sums", a.sum(), sp.sum())
            print("a\n", a)
            print("check\n", check)
            print("diff\n", check - a)
        else:
            print(dtype, sparse_cls, "ok")
