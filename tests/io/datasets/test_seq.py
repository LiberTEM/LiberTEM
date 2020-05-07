from libertem.executor.inline import InlineJobExecutor
from libertem.io.dataset.seq import SEQDataSet

def test_detect_unicode_error(default_raw, lt_ctx):
    path = default_raw._path
    SEQDataSet.detect_params(path, InlineJobExecutor())
