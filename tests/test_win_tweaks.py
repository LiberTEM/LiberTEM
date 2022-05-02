import sys
import builtins
import platform

import pytest


if platform.system() != 'Windows':
    pytest.skip("Skipping Windows-specific tests", allow_module_level=True)


real_import = builtins.__import__


def monkey_notfound(name, *args, **kwargs):
    if name in ('win32security', 'pywintypes'):
        raise ModuleNotFoundError(f"Mocked module not found {name}")
    return real_import(name, *args, **kwargs)


def monkey_importerror(name, *args, **kwargs):
    if name in ('win32security', ):
        raise ImportError(f"Mocked import error {name}")
    return real_import(name, *args, **kwargs)


def test_import_broken(monkeypatch):
    monkeypatch.delitem(sys.modules, 'win32security', raising=False)
    monkeypatch.delitem(sys.modules, 'libertem.common.win_tweaks', raising=False)
    monkeypatch.setattr(builtins, '__import__', monkey_importerror)

    # Self test
    with pytest.raises(ImportError):
        import win32security  # NOQA: F401

    from libertem.common.win_tweaks import get_owner_name

    # Make sure the dummy fallback version works
    # This would fail with the actual get_owner_name() function
    assert get_owner_name('/asdf/bla', None) == '<Unknown>'


def test_import_missing(monkeypatch):
    monkeypatch.delitem(sys.modules, 'win32security', raising=False)
    monkeypatch.delitem(sys.modules, 'libertem.common.win_tweaks', raising=False)
    monkeypatch.setattr(builtins, '__import__', monkey_notfound)

    # Self test
    with pytest.raises(ModuleNotFoundError):
        import win32security  # NOQA: F401

    from libertem.common.win_tweaks import get_owner_name

    # Make sure the dummy fallback version works
    # This would fail with the actual get_owner_name() function
    assert get_owner_name('/asdf/bla', None) == '<Unknown>'
