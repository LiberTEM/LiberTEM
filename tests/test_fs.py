import os
import uuid
import pickle

import pytest

from libertem.io.fs import _get_alt_path, FSError


@pytest.mark.skipif(os.name != 'nt',
                    reason="only runs on windows")
def test_doesnt_exist_windows():
    rnd = str(uuid.uuid4())
    path = r"C:\\" + rnd + r"\really\this\does\not\exist\\"

    alt = str(_get_alt_path(path))
    assert alt == "C:\\"


@pytest.mark.skipif(os.name == 'nt',
                    reason="doesnt run on windows")
def test_doesnt_exist_posix():
    rnd = str(uuid.uuid4())
    path = "/%s/really/this/doesnt/exist/" % rnd

    alt = str(_get_alt_path(path))
    assert alt == "/"


def test_pickle_fserror():
    e = FSError("wat", code="WAT", alternative="/wat/")
    pickled = pickle.dumps(e)
    pickle.loads(pickled)
