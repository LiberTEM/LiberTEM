import pytest
from libertem.executor.inline import InlineJobExecutor
from libertem import api as lt


@pytest.fixture
def inline_executor():
    return InlineJobExecutor()


@pytest.fixture
def lt_ctx(inline_executor):
    return lt.Context(executor=inline_executor)
