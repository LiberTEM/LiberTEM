import numpy as np

# Import here for backwards compatibility, refs #1031
from libertem.common.analysis import AnalysisResult, AnalysisResultSet, Analysis  # NOQA: F401


class BaseAnalysis(Analysis):
    TYPE = 'UDF'

    def __init__(self, dataset, parameters):
        self.dataset = dataset
        self.parameters = self.get_parameters(parameters)
        self.parameters.update(parameters)

        if self.TYPE == 'JOB':
            raise RuntimeError("Job support was removed in 0.7")

    def get_roi(self):
        return None

    def get_complex_results(
            self, job_result, key_prefix, title, desc, damage, default_lin=True):
        from libertem.viz import visualize_simple, CMAP_CIRCULAR_DEFAULT
        damage = damage & np.isfinite(job_result)
        magn = np.abs(job_result)
        angle = np.angle(job_result)
        wheel = CMAP_CIRCULAR_DEFAULT.rgb_from_vector(
            (job_result.real, job_result.imag, 0),
            vmax=np.max(magn[damage])
        )
        return [
            # for compatability, the magnitude has key=key_prefix
            AnalysisResult(
                raw_data=magn,
                visualized=visualize_simple(magn, damage=damage),
                key=key_prefix if default_lin else f'{key_prefix}_lin',
                title="%s [magn]" % title,
                desc="%s [magn]" % desc,
            ),
            AnalysisResult(
                raw_data=magn,
                visualized=visualize_simple(magn, logarithmic=True, damage=damage),
                key=f'{key_prefix}_log' if default_lin else key_prefix,
                title="%s [log(magn)]" % title,
                desc="%s [log(magn)]" % desc,
            ),
            AnalysisResult(
                raw_data=job_result.real,
                visualized=visualize_simple(job_result.real, damage=damage),
                key="%s_real" % key_prefix,
                title="%s [real]" % title,
                desc="%s [real]" % desc,
            ),
            AnalysisResult(
                raw_data=job_result.imag,
                visualized=visualize_simple(job_result.imag, damage=damage),
                key="%s_imag" % key_prefix,
                title="%s [imag]" % title,
                desc="%s [imag]" % desc,
            ),
            AnalysisResult(
                raw_data=angle,
                visualized=visualize_simple(angle, damage=damage),
                key="%s_angle" % key_prefix,
                title="%s [angle]" % title,
                desc="%s [angle]" % desc,
            ),
            AnalysisResult(
                raw_data=job_result,
                visualized=wheel,
                key="%s_complex" % key_prefix,
                title="%s [complex]" % title,
                desc="%s [complex]" % desc,
            ),
        ]

    def get_parameters(self, parameters):
        """
        Get analysis parameters. Override to set defaults
        """
        return parameters
