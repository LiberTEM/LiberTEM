import sys
import builtins
import platform

import pytest


if platform.system() != 'Windows':
    pytest.skip("Skipping Windows-specific tests")


real_import = builtins.__import__


def monkey_import_notfound(name, globals=None, locals=None, fromlist=(), level=0):
    if name in ('win32security', 'pywintypes'):
        raise ModuleNotFoundError(f"Mocked module not found {name}")
    return real_import(name, globals=globals, locals=locals, fromlist=fromlist, level=level)


def monkey_import_importerror(name, globals=None, locals=None, fromlist=(), level=0):
    if name in ('win32security', ):
        raise ImportError(f"Mocked import error {name}")
    return real_import(name, globals=globals, locals=locals, fromlist=fromlist, level=level)


def test_import_selftest(monkeypatch):
    monkeypatch.delitem(sys.modules, 'win32security', raising=False)
    monkeypatch.setattr(builtins, '__import__', monkey_import_importerror)

    with pytest.raises(ImportError):
        import win32security


def test_import_selftest2(monkeypatch):
    monkeypatch.delitem(sys.modules, 'win32security', raising=False)
    monkeypatch.setattr(builtins, '__import__', monkey_import_notfound)

    with pytest.raises(ModuleNotFoundError):
        import win32security


def test_import_broken(monkeypatch):
    monkeypatch.delitem(sys.modules, 'win32security', raising=False)
    monkeypatch.delitem(sys.modules, 'libertem.win_tweaks', raising=False)
    monkeypatch.setattr(builtins, '__import__', monkey_import_importerror)

    from libertem.win_tweaks import get_owner_name

    # Make sure the dummy fallback version works
    # This would fail with the actual get_owner_name() function
    assert get_owner_name('/asdf/bla', None) == '<Unknown>'


def test_import_missing(monkeypatch):
    monkeypatch.delitem(sys.modules, 'win32security', raising=False)
    monkeypatch.delitem(sys.modules, 'libertem.win_tweaks', raising=False)
    monkeypatch.setattr(builtins, '__import__', monkey_import_notfound)

    from libertem.win_tweaks import get_owner_name

    # Make sure the dummy fallback version works
    # This would fail with the actual get_owner_name() function
    assert get_owner_name('/asdf/bla', None) == '<Unknown>'
