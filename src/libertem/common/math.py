try:
    from math import prod
except ImportError:
    def prod(iterable):
        '''
        Safe product for large size calculations.

        :meth:`numpy.prod` uses 32 bit for default :code:`int` on Windows 64 bit. This
        function is a replacement for :meth:`math.prod` for versions < Python 3.8
        that uses infinite width integers.
        '''
        result = 1
        for item in iterable:
            result *= item
        return result
