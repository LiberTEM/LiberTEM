def test_ctx_load(lt_ctx, default_raw):
    lt_ctx.load(
        "raw",
        path=default_raw._path,
        scan_size=(16, 16),
        dtype="float32",
        detector_size=(128, 128),
    )
