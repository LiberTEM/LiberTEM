import os

from libertem.io.utils import get_owner_name


def test_get_owner_name():
    owner = get_owner_name(__file__, os.stat(__file__))
    assert isinstance(owner, str)
    assert len(owner) > 0
