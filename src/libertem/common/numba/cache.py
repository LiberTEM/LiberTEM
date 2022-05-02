import time
import hashlib
import logging

logger = logging.getLogger(__name__)

import numba  # NOQA: E402
try:
    from numba.core.serialize import dumps
    from numba.core.caching import FunctionCache
    from numba.core.registry import dispatcher_registry, CPUDispatcher
    old_dispatch = True
except ImportError:
    try:
        # numba 0.54 changes the internal structure, so we have to adapt our imports:
        from numba.core.serialize import dumps
        from numba.core.caching import FunctionCache
        from numba.core.registry import CPUDispatcher
        from numba.core.target_extension import (
            dispatcher_registry, jit_registry, target_registry, Generic
        )
        old_dispatch = False
    except ImportError as e:
        CPUDispatcher = None
        dispatcher_registry = None
        logger.warning(
            "could not register custom numba dispatcher, disabling custom cache (%s)" % str(e)
        )
        logger.warning(
            "numba version %s" % str(numba.__version__)
        )


_cached_njit_reg = []


def cached_njit(*args, **kwargs):
    """
    Replacement for numba.njit with custom caching. Only supports usage
    with parameters, i.e.

    @cached_njit()
    def fn():
        ...
    """
    def wrapper(fn):
        # only register with the custom target if we manage to import the right
        # structures from numba:
        if dispatcher_registry is None:
            kwargs.update({'cache': True})
        else:
            kwargs.update({'_target': 'custom_cpu', 'cache': True})
            _cached_njit_reg.append(fn)
        return numba.njit(fn, *args, **kwargs)
    return wrapper


def hasher(x):
    return hashlib.sha256(x).hexdigest()


if dispatcher_registry is not None:
    class MyFunctionCache(FunctionCache):
        def _get_dependencies(self, cvar):
            deps = [cvar]
            if hasattr(cvar, 'py_func'):
                # TODO: does the cache key need to depend on any other
                # attributes of the Dispatcher?
                closure = cvar.py_func.__closure__
                deps = [cvar.py_func.__code__.co_code]
            elif hasattr(cvar, '__closure__'):
                closure = cvar.__closure__
                # if cvar is a function and closes over a Dispatcher, the
                # cache will be busted because of the uuid that is regenerated
                deps = [cvar.__code__.co_code]
            else:
                closure = None
            if closure is not None:
                for x in closure:
                    deps.extend(self._get_dependencies(x.cell_contents))
            return deps

        def _index_key(self, sig, codegen):
            """
            Compute index key for the given signature and codegen.
            It includes a description of the OS, target architecture and hashes of
            the bytecode for the function and, if the function has a __closure__,
            a hash of the cell_contents.
            """
            codebytes = self._py_func.__code__.co_code
            cvars = self._get_dependencies(self._py_func)
            if len(cvars) > 0:
                cvarbytes = dumps(cvars)
            else:
                cvarbytes = b''

            return (sig, "libertem-numba-cache", codegen.magic_tuple(), (hasher(codebytes),
                                                 hasher(cvarbytes),))

        def load_overload(self, *args, **kwargs):
            t0 = time.time()
            data = super().load_overload(*args, **kwargs)
            t1 = time.time()
            if data is None:
                logger.info(f"numba cache miss {self._name} {self._py_func}")
            else:
                logger.info(f"cache hit for {self._name}, load took {(t1 - t0):.3f}s")
            return data

    # if we can hack in our custom caching, do it:
    class MyCPUDispatcher(CPUDispatcher):
        def enable_caching(self):
            self._cache = MyFunctionCache(self.py_func)

    if old_dispatch:
        dispatcher_registry['custom_cpu'] = MyCPUDispatcher
    else:
        class MyCPU(Generic):
            pass

        dispatcher_registry[MyCPU] = MyCPUDispatcher
        jit_registry[MyCPU] = cached_njit  # FIXME: is this needed?
        target_registry["custom_cpu"] = MyCPU
