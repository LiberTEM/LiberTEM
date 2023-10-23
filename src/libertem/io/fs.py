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


def _access_ok(path: Path):
    try:
        # May trigger PermissionError on Windows
        allowed = os.path.isdir(path) and os.access(path, os.R_OK | os.X_OK)
        if allowed:
            Path(path).resolve()
            # Windows sometimes raises an exception even if "allowed"
            os.listdir(path)
        return allowed
    except PermissionError:
        return False


def _get_alt_path(path: str):
    cur_path = Path(path).expanduser()
    try:
        cur_path = cur_path.resolve()
    # Triggered by empty DVD drive or permission denied on Windows
    except PermissionError:
        # we just skip resolving at this point and
        # move one up
        pass
    while not _access_ok(cur_path):
        try:
            cur_path = cur_path.parents[0]
            cur_path = cur_path.resolve()
        except IndexError:  # from cur_path.parents[0]
            # There is no parent
            return Path.home()
        except PermissionError:  # from cur_path.resolve()
            # No access yet, we move one up again
            pass
    return cur_path


def stat_path(path: str):
    try:
        return Path(path).expanduser().resolve().stat()
    except FileNotFoundError:
        raise FSError(
            code="NOT_FOUND",
            msg="path %s could not be found" % path,
            alternative=str(_get_alt_path(path)),
        )
    except PermissionError as e:
        raise FSError(
            code="PERMISSION_ERROR",
            msg=str(e),
            alternative=str(_get_alt_path(path)),
        )


def get_fs_listing(path: str):
    try:
        abspath = Path(path).expanduser().resolve()
    # Triggered by empty DVD drive on Windows
    except PermissionError as e:
        raise FSError(
            code="PERMISSION_ERROR",
            msg=str(e),
            alternative=str(_get_alt_path(path)),
        )
    if not abspath.is_dir():
        raise FSError(
            code="NOT_FOUND",
            msg="path %s could not be found" % path,
            alternative=str(_get_alt_path(path)),
        )
    if not _access_ok(abspath):
        raise FSError(
            code="ACCESS_DENIED",
            msg="access to %s was denied" % path,
            alternative=str(_get_alt_path(path)),
        )
    names = os.listdir(abspath)
    dirs = []
    files = []
    names = [".."] + names
    for name in names:
        full_path = os.path.join(abspath, name)
        try:
            s = os.stat(full_path)
        except (FileNotFoundError, PermissionError):
            # this can happen either because of a TOCTOU-like race condition
            # or for example for things like broken softlinks
            continue

        try:
            owner = get_owner_name(full_path, s)
        except FileNotFoundError:  # only from win_tweaks.py version
            continue
        except OSError:  # only from win_tweaks.py version
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
        "path": str(abspath),
        "files": files,
        "dirs": dirs,
        "drives": drives,
        "places": places,
    }
