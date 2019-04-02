import sys
import msvcrt
import ctypes
from ctypes import windll, wintypes, byref

import win32security
import pywintypes


def get_owner_name(full_path, stat):
    try:
        s = win32security.GetFileSecurity(full_path, win32security.OWNER_SECURITY_INFORMATION)
        sid = s.GetSecurityDescriptorOwner()
        (name, domain, t) = win32security.LookupAccountSid(None, sid)
        return "%s\\%s" % (domain, name)
    except pywintypes.error as e:
        raise IOError(e)


ENABLE_QUICK_EDIT = 0x0040
ENABLE_EXTENDED_FLAGS = 0x0080


def get_console_mode(stream=sys.stdin):
    file_handle = msvcrt.get_osfhandle(stream.fileno())
    getConsoleMode = windll.kernel32.GetConsoleMode
    getConsoleMode.argtypes, getConsoleMode.restype = ([wintypes.HANDLE, wintypes.LPDWORD],
                                                       wintypes.BOOL)
    mode = wintypes.DWORD(0)
    if getConsoleMode(file_handle, byref(mode)):
        return mode.value
    else:
        err = ctypes.get_last_error()
        raise ctypes.WinError(err)


def set_console_mode(mode, stream=sys.stdin):
    file_handle = msvcrt.get_osfhandle(stream.fileno())
    setConsoleMode = windll.kernel32.SetConsoleMode
    setConsoleMode.argtypes, setConsoleMode.restype = ([wintypes.HANDLE, wintypes.DWORD],
                                                       wintypes.BOOL)
    if setConsoleMode(file_handle, mode):
        return
    else:
        err = ctypes.get_last_error()
        raise ctypes.WinError(err)


def disable_quickedit():
    mode = get_console_mode()
    mode &= ~ENABLE_QUICK_EDIT
    mode |= ENABLE_EXTENDED_FLAGS
    set_console_mode(mode)


if __name__ == "__main__":
    disable_quickedit()
