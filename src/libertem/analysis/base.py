from typing import (
    Optional, TYPE_CHECKING
)
from typing_extensions import Literal

import numpy as np

from libertem.io.dataset.base import DataSet
from libertem.udf.base import UDF, UDFResultDict

# Base classes for results moved to common for MIT licensing, refs #1031
from libertem.common.analysis import AnalysisResult, AnalysisResultSet

if TYPE_CHECKING:
    from libertem.analysis.helper import GeneratorHelper
    from libertem.web.rpc import ProcedureProtocol
    import numpy.typing as nt


class Analysis:
    """
    Abstract base class for Analysis classes.

    An Analysis is the interface between a UDF and the Web API, and handles
    visualization of partial and full results.

    Passing an instance of an :class:`Analysis` sub-class to
    :meth:`libertem.api.Context.run` will generate an :class:`AnalysisResultSet`.
    The content of this result set is governed by the specific implementation of
    the :code:`Analysis` sub-class.

    .. versionadded:: 0.3.0

    .. versionchanged:: 0.7.0
        Removed deprecated methods :code:`get_results` and :code:`get_job`
    """
    TYPE: Literal["UDF"] = "UDF"
    registry: dict[str, "type[Analysis]"] = {}

    def __init__(self, dataset: DataSet, parameters: dict):
        self.dataset = dataset

    def __init_subclass__(cls, id_=None, **kwargs):

        # override id_ with your own id
        # Used to register the subclass
        # https://www.python.org/dev/peps/pep-0487/#subclass-registration

        super().__init_subclass__(**kwargs)
        if id_ is not None:
            cls.registry[id_] = cls

    @classmethod
    def get_analysis_by_type(cls, id_: str) -> type["Analysis"]:
        return cls.registry[id_]

    @classmethod
    def get_template_helper(cls) -> type["GeneratorHelper"]:
        raise NotImplementedError()

    @classmethod
    def get_rpc_definitions(cls) -> dict[str, type["ProcedureProtocol"]]:
        return {}

    async def controller(self, cancel_id, executor, job_is_cancelled, send_results):
        raise NotImplementedError()

    def get_udf_results(
        self, udf_results: UDFResultDict, roi: Optional[np.ndarray],
        damage: "nt.ArrayLike",
    ) -> AnalysisResultSet:
        """
        Convert UDF results to a :code:`AnalysisResultSet`,
        including visualizations.

        Parameters
        ----------
        udf_results
            raw results from the UDF
        roi : numpy.ndarray or None
            Boolean array of the navigation dimension

        Returns
        -------
        list of AnalysisResult
            one or more annotated results
        """
        raise NotImplementedError()

    def get_udf(self) -> UDF:
        """
        Set TYPE='UDF' on the class and implement this method to run a UDF
        from this analysis
        """
        raise NotImplementedError()

    def get_roi(self) -> Optional[np.ndarray]:
        """
        Get the region of interest the UDF should be run on. For example,
        the parameters could describe some geometry, which this method should
        convert to a boolean array. See also: :func:`libertem.analysis.getroi.get_roi`

        Returns
        -------
        numpy.ndarray or None
            region of interest for which we want to run our analysis
        """
        raise NotImplementedError()

    def get_complex_results(self, job_result, key_prefix, title, desc, damage, default_lin=True):
        raise NotImplementedError()

    def get_parameters(self, parameters: dict) -> dict:
        """
        Get analysis parameters. Override to set defaults
        """
        raise NotImplementedError()

    def need_rerun(self, old_params: dict, new_params: dict) -> bool:
        """
        Determine if the analysis needs to be re-run on the data. If not,
        we can just call `get_udf_results` again, for example if the parameters
        only change the visualization.

        Parameters
        ----------
        old_params : Dict
        new_params : Dict

        Returns
        -------
        bool
            True iff the parameter change needs to cause a re-run on the data
        """
        return True


class BaseAnalysis(Analysis):
    def __init__(self, dataset, parameters):
        super().__init__(dataset, parameters)
        self.parameters = self.get_parameters(parameters)
        self.parameters.update(parameters)

        if self.TYPE == 'JOB':
            raise RuntimeError("Job support was removed in 0.7")

    def get_roi(self):
        return None

    def get_complex_results(
            self, job_result, key_prefix, title, desc, damage, default_lin=True):
        from libertem.viz import visualize_simple, rgb_from_2dvector
        damage = damage & np.isfinite(job_result)
        magn = np.abs(job_result)
        angle = np.angle(job_result)
        wheel = rgb_from_2dvector(
            x=job_result.real,
            y=job_result.imag,
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

    def get_parameters(self, parameters: dict):
        """
        Get analysis parameters. Override to set defaults
        """
        return parameters


__all__ = ['AnalysisResult', 'AnalysisResultSet', 'Analysis', 'BaseAnalysis']
