from libertem.analysis import (
    DiskMaskAnalysis, RingMaskAnalysis, PointMaskAnalysis,
    FEMAnalysis, COMAnalysis, SumAnalysis, PickFrameAnalysis,
    PickFFTFrameAnalysis, SumfftAnalysis,
    RadialFourierAnalysis, ApplyFFTMask, SDAnalysis, SumSigAnalysis, ClusterAnalysis
)


class GeneratorHelper:

    def __init__(self):
        self.short_name = None
        self.api = None

    def convert_params(self, raw_params, ds):
        return None

    def get_plot(self):
        return None


# same function inside job
def get_analysis_by_type(type_):

    analysis_by_type = {
        "APPLY_DISK_MASK": DiskMaskAnalysis,
        "APPLY_RING_MASK": RingMaskAnalysis,
        "FFTSUM_FRAMES": SumfftAnalysis,
        "APPLY_POINT_SELECTOR": PointMaskAnalysis,
        "CENTER_OF_MASS": COMAnalysis,
        "RADIAL_FOURIER": RadialFourierAnalysis,
        "SUM_FRAMES": SumAnalysis,
        "PICK_FRAME": PickFrameAnalysis,
        "FEM": FEMAnalysis,
        "PICK_FFT_FRAME": PickFFTFrameAnalysis,
        "APPLY_FFT_MASK": ApplyFFTMask,
        "SD_FRAMES": SDAnalysis,
        "SUM_SIG": SumSigAnalysis,
        "CLUST": ClusterAnalysis
    }
    return analysis_by_type[type_]
