import os

import numpy as np
import pytest

from libertem.udf.base import UDF


class LargeParamUDF(UDF):
    '''
    This UDF can compare the efficiency of transferring
    large parameters through the regular parameter interface
    or through a memory-mapped file.

    This is useful to benchmark the efficiency of parameter handling in
    executor and UDFRunner implementations

    Parameters
    ----------

    A : str or numpy.ndarray
        If it is of type :code:`str`, the UDF uses this as a filename
        for :code:`np.load(..., mmap_mode='r')` as a simple
        and portable method for shared memory.
    '''
    def __init__(self, A):
        super().__init__(A=A)

    def get_task_data(self):
        if isinstance(self.params.A, str):
            A = np.load(self.params.A, mmap_mode='r')
        else:
            A = self.params.A
        return {
            'A': A
        }

    def process_tile(self, tile):
        # access the data
        np.sum(self.task_data.A)

    def postprocess(self):
        # Make sure the file can be overwritten on Windows
        # by closing the file handle
        if hasattr(self.task_data.A, '_mmap'):
            self.task_data.A._mmap.close()

    def get_result_buffers(self):
        return {}


class CheatResultUDF(UDF):
    def __init__(self, result_folder):
        self._cheat_result = None
        super().__init__(result_folder=result_folder)

    def get_result_buffers(self):
        return {
            'result': self.buffer(
                kind='single',
                extra_shape=(1024*1024, ),
                dtype=np.float32,
                use='result_only'
            ),
            'result_files': self.buffer(
                kind='single',
                dtype=object,
                use='private'
            ),
        }

    def preprocess(self):
        self.results.result_files[0] = []

    def process_partition(self, partition):
        count = len(partition)
        result = np.full(1024*1024, count, dtype=np.float32)
        filename = str(self.meta.slice.origin) + '.npy'
        path = os.path.join(self.params.result_folder, filename)
        np.save(path, result)
        self.results.result_files[0].append(path)

    def merge(self, dest, src):
        if self._cheat_result is None:
            path = src.result_files[0][0]
            template = np.load(path, mmap_mode='r')
            self._cheat_result = np.zeros_like(template)
            template._mmap.close()
        for path in src.result_files[0]:
            result = np.load(path, mmap_mode='r')
            self._cheat_result += result
            result._mmap.close()
            os.remove(path)

    def get_results(self):
        return {
            'result': self._cheat_result
        }


class LargeResultUDF(UDF):

    def get_result_buffers(self):
        return {
            'result': self.buffer(
                kind='sig',
                extra_shape=(1024*1024, ),
                dtype=np.float32,
            ),
        }

    def process_partition(self, partition):
        count = len(partition)
        self.results.result += count

    def merge(self, dest, src):
        dest.result += src.result


# class to share dist ctx
class Test:
    @pytest.mark.benchmark(
        group="udf parameters"
    )
    @pytest.mark.parametrize(
        'method', ('preshared_file', 'file', 'executor')
    )
    def test_param(self, shared_dist_ctx, benchmark, tmp_path, method):
        data = np.zeros(1024*1024, dtype=np.float32)

        ds = shared_dist_ctx.load(
            'memory',
            data=np.zeros((1024, 2)),
            sig_dims=1,
            num_partitions=32
        )

        if method == 'preshared_file':
            A = str(tmp_path / 'A.npy')
            np.save(A, data)

            def benchfun():
                shared_dist_ctx.run_udf(dataset=ds, udf=LargeParamUDF(A=A))

        elif method == 'file':
            def benchfun():
                A = str(tmp_path / 'A.npy')
                np.save(A, data)
                shared_dist_ctx.run_udf(dataset=ds, udf=LargeParamUDF(A=A))

        else:
            A = data

            def benchfun():
                shared_dist_ctx.run_udf(dataset=ds, udf=LargeParamUDF(A=A))

        benchmark(benchfun)

    @pytest.mark.benchmark(
        group="udf results"
    )
    @pytest.mark.parametrize(
        'method', ('file', 'executor')
    )
    def test_result(self, shared_dist_ctx, benchmark, tmp_path, method):
        if method == 'file':
            udf = CheatResultUDF(str(tmp_path))
        else:
            udf = LargeResultUDF()

        ds = shared_dist_ctx.load(
            'memory',
            data=np.zeros((1024, 2)),
            sig_dims=1,
            num_partitions=32
        )

        benchmark(
            shared_dist_ctx.run_udf,
            dataset=ds,
            udf=udf
        )
        if hasattr(udf, '_cheat_result'):
            udf._cheat_result[:] = 0
        # Validate that it actually did the job
        res = shared_dist_ctx.run_udf(dataset=ds, udf=udf)
        assert np.allclose(res['result'].raw_data, 1024)
