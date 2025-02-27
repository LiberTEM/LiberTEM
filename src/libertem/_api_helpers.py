import logging
from typing import Any
from collections.abc import Generator

from libertem.udf.base import UDFResults, UDFRunner

logger = logging.getLogger(__name__)


class ResultGenerator:
    """
    Yields partial results from one or more UDFs currently running,
    with methods to control some aspects of the UDF run.
    """

    def __init__(
        self,
        task_results: Generator[UDFResults, None, None],
        runner: UDFRunner,
        result_iter,
    ):
        self._task_results = task_results
        self._runner = runner
        self._result_iter = result_iter

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._task_results)

    def close(self):
        self._task_results.close()
        self._result_iter.close()

    def update_parameters_experimental(self, parameters: list[dict[str, Any]]):
        """
        Update parameters while the UDFs are running.

        :code:`parameters` should be a list of dicts, with one dict for each
        UDF you are running.

        The dicts should only contain items for those parameters you want to
        update.
        """
        logger.debug("ResultGenerator.update_parameters_experimental: %s", parameters)
        self._result_iter.update_parameters_experimental(parameters)

    def throw(self, exc: Exception):
        return self._result_iter.throw(exc)


class ResultAsyncGenerator:
    """
    async wrapper of `ResultGenerator`.
    """

    def __init__(self, result_generator: ResultGenerator):
        from concurrent.futures import ThreadPoolExecutor
        self._result_generator = result_generator
        self._pool = ThreadPoolExecutor(max_workers=1)

    def __aiter__(self):
        return self

    async def __anext__(self):
        from .common.async_utils import MyStopIteration
        try:
            return await self._run_in_executor(self._inner_next)
        except MyStopIteration:
            raise StopAsyncIteration()

    def _inner_next(self):
        from .common.async_utils import MyStopIteration
        try:
            return next(self._result_generator)
        except StopIteration:
            raise MyStopIteration()

    async def aclose(self):
        return await self._run_in_executor(self._result_generator.close)

    async def _run_in_executor(self, f, *args):
        import asyncio
        loop = asyncio.get_event_loop()
        next_result = await loop.run_in_executor(self._pool, f, *args)
        return next_result

    async def update_parameters_experimental(self, parameters: list[dict[str, Any]]):
        """
        Update parameters while the UDFs are running.

        :code:`parameters` should be a list of dicts, with one dict for each
        UDF you are running.

        The dicts should only contain items for those parameters you want to
        update.
        """
        logger.debug("ResultGenerator.update_parameters_experimental: %s", parameters)
        return await self._run_in_executor(
            self._result_generator.update_parameters_experimental, parameters
        )

    async def athrow(self, exc: Exception):
        return await self._run_in_executor(self._result_generator.throw, exc)


class UDFAsyncRun:
    """
    Helper object that is returned from `Context.run_udf_async`
    """
    def __init__(self, sgen):
        self._sgen = sgen

    async def updates(self):
        yield None

    async def get_latest(self):
        return None
