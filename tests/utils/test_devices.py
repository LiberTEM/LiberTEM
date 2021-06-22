import sys
import builtins

import pytest


real_import = builtins.__import__


def monkey_notfound(name, *args, **kwargs):
    if name in ('cupy', ):
        raise ModuleNotFoundError(f"Mocked module not found {name}")
    return real_import(name, *args, **kwargs)


def monkey_importerror(name, *args, **kwargs):
    if name in ('cupy', ):
        raise ImportError(f"Mocked import error {name}")
    return real_import(name, *args, **kwargs)


def monkey_brokencupycuda(name, *args, **kwargs):
    if name in ('cupy.cuda', ):
        raise AttributeError(f"Mocked import error {name}")
    return real_import(name, *args, **kwargs)


def test_detect_error(monkeypatch):
    monkeypatch.delitem(sys.modules, 'cupy', raising=False)
    monkeypatch.delitem(sys.modules, 'cupy.cuda', raising=False)
    monkeypatch.setattr(builtins, '__import__', monkey_importerror)

    # Self test
    with pytest.raises(ImportError):
        import cupy  # NOQA: F401

    monkeypatch.delitem(sys.modules, 'libertem.utils.devices', raising=False)

    with pytest.warns(RuntimeWarning, match="Mocked import error cupy"):
        from libertem.utils.devices import detect
        result = detect()

    assert not result['has_cupy']


def test_detect_notfound(monkeypatch):
    monkeypatch.delitem(sys.modules, 'cupy', raising=False)
    monkeypatch.delitem(sys.modules, 'cupy.cuda', raising=False)
    monkeypatch.setattr(builtins, '__import__', monkey_notfound)

    # Self test
    with pytest.raises(ModuleNotFoundError):
        import cupy  # NOQA: F401

    monkeypatch.delitem(sys.modules, 'libertem.utils.devices', raising=False)

    from libertem.utils.devices import detect
    result = detect()

    assert not result['has_cupy']


def test_detect_nocuda(monkeypatch):
    try:
        import cupy
    except Exception:
        pytest.skip("Importable CuPy required for this test.")
    monkeypatch.delattr(sys.modules['cupy'], 'cuda', raising=False)
    # Self test
    with pytest.raises(Exception):
        cupy.cuda

    monkeypatch.delitem(sys.modules, 'libertem.utils.devices', raising=False)

    with pytest.warns(RuntimeWarning, match="module 'cupy' has no attribute 'cuda'"):
        from libertem.utils.devices import detect
        result = detect()

    assert not result['has_cupy']


def test_detect_broken(monkeypatch):
    try:
        import cupy
    except Exception:
        pytest.skip("Importable CuPy required for this test.")

    # CuPy can throw all kinds of exceptions, depending on what exactly goes wrong
    class FunkyException(Exception):
        pass

    def badfunc(*args, **kwargs):
        raise FunkyException()

    monkeypatch.setattr(cupy, 'array', badfunc)
    monkeypatch.setattr(cupy, 'zeros', badfunc)

    # Self test
    with pytest.raises(FunkyException):
        cupy.array(cupy.zeros(1,))

    monkeypatch.delitem(sys.modules, 'libertem.utils.devices', raising=False)

    from libertem.utils.devices import detect
    with pytest.warns(RuntimeWarning, match='FunkyException'):
        result = detect()

    assert not result['has_cupy']
