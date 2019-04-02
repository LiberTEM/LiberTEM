import os
import stat
from pathlib import Path

import psutil

from libertem.io.utils import get_owner_name


class FSError(Exception):
    def __init__(self, msg, code, alternative=None):
        super().__init__(msg, code, alternative)
        self.code = code
        self.alternative = alternative


def _access_ok(path):
    return os.path.isdir(path) and os.access(path, os.R_OK | os.X_OK)


def _get_alt_path(path):
    cur_path = Path(path).resolve()
    while not _access_ok(cur_path):
        cur_path = cur_path / '..'
        cur_path = cur_path.resolve()
        # we have reached the root (either / or a drive letter on windows)
        if Path(cur_path.anchor) == cur_path:
            if _access_ok(cur_path):
                return cur_path
            else:
                # we can only suggest the home directory:
                return str(Path.home())
    return cur_path


def get_fs_listing(path):
    if not os.path.isdir(path):
        raise FSError(
            code="NOT_FOUND",
            msg="path %s could not be found" % path,
            alternative=str(_get_alt_path(path)),
        )
    if not _access_ok(path):
        raise FSError(
            code="ACCESS_DENIED",
            msg="access to %s was denied" % path,
            alternative=str(_get_alt_path(path)),
        )
    path = os.path.abspath(path)
    names = os.listdir(path)
    dirs = []
    files = []
    names = [".."] + names
    for name in names:
        full_path = os.path.join(path, name)
        try:
            s = os.stat(full_path)
        except FileNotFoundError:
            # this can happen either because of a TOCTOU-like race condition
            # or for example for things like broken softlinks
            continue

        try:
            owner = get_owner_name(full_path, s)
        except FileNotFoundError:  # only from win_tweaks.py version
            continue
        except IOError:  # only from win_tweaks.py version
            owner = "<Unknown>"

        res = {"name": name, "stat": s, "owner": owner}
        if stat.S_ISDIR(s.st_mode):
            dirs.append(res)
        else:
            files.append(res)
    drives = [
        part.mountpoint
        for part in psutil.disk_partitions()
        if part.fstype != "squashfs"
    ]
    places = [
        {"key": "home", "title": "Home", "path": str(Path.home())},
    ]

    return {
        "path": path,
        "files": files,
        "dirs": dirs,
        "drives": drives,
        "places": places,
    }
