from libertem.executor.inline import InlineJobExecutor
from libertem.io.dataset.seq import SEQDataSet

def test_detect_unicode_error(raw_with_zeros, lt_ctx):
    path = raw_with_zeros._path
    SEQDataSet.detect_params(path, InlineJobExecutor())
