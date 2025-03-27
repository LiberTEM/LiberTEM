from numpy.testing import assert_allclose
from libertem.analysis.disk import DiskMaskAnalysis


def test_compare_hdf5_result_with_raw_result(
    lt_ctx, hdf5_same_data_4d, raw_same_dataset_4d
):
    ds_hdf5 = lt_ctx.load(
        "hdf5", path=hdf5_same_data_4d.filename, ds_path="data",
    )
    analysis_hdf5 = DiskMaskAnalysis(
        dataset=ds_hdf5,
        parameters={"cx": 13, "cy": 13, "r": 13}
    )
    result_hdf5 = lt_ctx.run(analysis_hdf5)
    result_hdf5 = result_hdf5['intensity'].raw_data

    analysis_raw = DiskMaskAnalysis(
        dataset=raw_same_dataset_4d,
        parameters={"cx": 13, "cy": 13, "r": 13}
    )
    result_raw = lt_ctx.run(analysis_raw)
    result_raw = result_raw['intensity'].raw_data

    assert_allclose(result_hdf5, result_raw, rtol=1e-6, atol=1e-6)
