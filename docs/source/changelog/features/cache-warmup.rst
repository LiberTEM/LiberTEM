[Feature] Cache warmup when opening a data set
==============================================

* Precompiles jit-ed functions on a single process per node, in a controlled manner, preventing CPU oversubscription. This should further improve once numba can cache functions which capture other functions in their closure (:pr:`886`, :issue:`798`)
