import warnings
import pytest


def test_ctx_load(lt_ctx, default_raw):
    lt_ctx.load(
        "raw",
        path=default_raw._path,
        scan_size=(16, 16),
        dtype="float32",
        detector_size=(128, 128),
    )


def test_ctx_load_old(lt_ctx, default_raw):
    with warnings.catch_warnings(record=True) as w:
        lt_ctx.load(
            "raw",
            path=default_raw._path,
            scan_size=(16, 16),
            dtype="float32",
            detector_size_raw=(128, 128),
            crop_detector_to=(128, 128)
        )
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)


def test_missing_detector_size(lt_ctx, default_raw):
    with pytest.raises(ValueError) as e:
        lt_ctx.load(
            "raw",
            path=default_raw._path,
            scan_size=(16, 16),
            dtype="float32",
            )
    assert e.match("missing 1 required argument: 'detector_size'")
