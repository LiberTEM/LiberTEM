import os
from unittest import mock

import pytest

from libertem.io.fs import get_fs_listing


def test_get_fs_listing_permission_of_subdir(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp('get_fs_listing')
    tmpdir.mkdir('sub1')
    no_access_dir = tmpdir.mkdir('sub2')

    orig_stat = os.stat

    def mock_stat(path, *args, **kwargs):
        if os.path.normpath(path) == os.path.normpath(no_access_dir):
            raise PermissionError(f"[Errno 13] Access is denied: '{path}'")
        return orig_stat(path, *args, **kwargs)

    # patch os.stat to fail for `sub2`:
    with mock.patch('os.stat', side_effect=mock_stat):
        listing = get_fs_listing(tmpdir)

        dir_names = [
            entry['name']
            for entry in listing['dirs']
        ]
        assert 'sub2' not in dir_names

        # no direct access to `sub2`
        with pytest.raises(PermissionError):
            os.stat(no_access_dir)
