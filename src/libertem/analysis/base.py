import numpy as np

from libertem.viz import encode_image, visualize_simple, CMAP_CIRCULAR_DEFAULT


class AnalysisResult(object):
    """
    this class represents a single 2D image result
    """
    def __init__(self, raw_data, visualized, title, desc, key):
        self.raw_data = raw_data
        self._visualized = visualized
        self.title = title
        self.desc = desc
        self.key = key

    def __str__(self):
        result = ""
        for k in ("title", "desc", "key", "raw_data", "visualized"):
            result += "%s: %s\n" % (k, getattr(self, k))
        return result

    def __repr__(self):
        return "<AnalysisResult: %s>" % self.key

    def get_image(self, save_kwargs=None):
        return encode_image(self.visualized, save_kwargs=save_kwargs)

    @property
    def visualized(self):
        if callable(self._visualized):
            self._visualized = self._visualized()
        return self._visualized


class AnalysisResultSet(object):
    def __init__(self, results, raw_results=None):
        self._results = results
        self.raw_results = raw_results

    def __repr__(self):
        return repr(self.results)

    def __getattr__(self, k):
        for result in self.results:
            if result.key == k:
                return result
        raise AttributeError("result with key '%s' not found, have: %s" % (
            k, ", ".join([r.key for r in self.results])
        ))

    def __getitem__(self, k):
        if isinstance(k, str):
            return self.__getattr__(k)
        return self.results[k]

    def __len__(self):
        return len(self.results)

    @property
    def results(self):
        if callable(self._results):
            self._results = self._results()
        return self._results


class BaseAnalysis(object):
    TYPE = 'JOB'

    def __init__(self, dataset, parameters):
        self.dataset = dataset
        self.parameters = self.get_parameters(parameters)
        self.parameters.update(parameters)

    def get_results(self, job_results):
        """
        Parameters
        ----------
        job_results : list of :class:`~numpy.ndarray`
            raw results from the job

        Returns
        -------
        list of AnalysisResult
            one or more annotated results
        """
        raise NotImplementedError()

    def get_job(self):
        """
        Returns
        -------
        Job
            a Job instance
        """
        raise NotImplementedError()

    def get_udf(self):
        """
        set TYPE='UDF' on the class and implement this method to run a UDF
        from this analysis
        """
        raise NotImplementedError()

    def get_roi(self):
        """
        Returns
        -------
        numpy.ndarray or None
            region of interest for which we want to run our analysis
        """
        return None

    def get_complex_results(self, job_result, key_prefix, title, desc):
        magn = np.abs(job_result)
        angle = np.angle(job_result)
        wheel = CMAP_CIRCULAR_DEFAULT.rgb_from_vector((job_result.imag, job_result.real))
        return [
            # for compatability, the magnitude has key=key_prefix
            AnalysisResult(
                raw_data=magn,
                visualized=visualize_simple(magn),
                key=key_prefix,
                title="%s [magn]" % title,
                desc="%s [magn]" % desc,
            ),
            AnalysisResult(
                raw_data=job_result.real,
                visualized=visualize_simple(job_result.real),
                key="%s_real" % key_prefix,
                title="%s [real]" % title,
                desc="%s [real]" % desc,
            ),
            AnalysisResult(
                raw_data=job_result.imag,
                visualized=visualize_simple(job_result.imag),
                key="%s_imag" % key_prefix,
                title="%s [imag]" % title,
                desc="%s [imag]" % desc,
            ),
            AnalysisResult(
                raw_data=angle,
                visualized=visualize_simple(angle),
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
