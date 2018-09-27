from libertem.viz import encode_image


class AnalysisResult(object):
    """
    this class represents a single 2D image result
    """
    def __init__(self, raw_data, visualized, title, desc):
        self.raw_data = raw_data
        self.visualized = visualized
        self.title = title
        self.desc = desc

    def __str__(self):
        return self.title

    def __repr__(self):
        return "<AnalysisResult: %s>" % self.title

    def get_image(self, save_kwargs=None):
        return encode_image(self.visualized, save_kwargs=save_kwargs)


class BaseAnalysis(object):
    def __init__(self, dataset, parameters):
        self.dataset = dataset
        self.parameters = parameters

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
