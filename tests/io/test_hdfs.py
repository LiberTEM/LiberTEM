import pytest

try:
    from libertem.io.dataset.hdfs import BinaryHDFSDataSet
    HAVE_HDFS = True
except ImportError:
    HAVE_HDFS = False

pytestmark = pytest.mark.skipif(not HAVE_HDFS, reason="need HDFS instance")  # NOQA


def test_dataset_is_picklable():
    pass  # TODO
