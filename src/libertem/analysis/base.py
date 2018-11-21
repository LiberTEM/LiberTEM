from libertem.viz import encode_image


class AnalysisResult(object):
    """
    this class represents a single 2D image result
    """
    def __init__(self, raw_data, visualized, title, desc, key):
        self.raw_data = raw_data
        self.visualized = visualized
        self.title = title
        self.desc = desc
        self.key = key

    def __str__(self):
        return self.title

    def __repr__(self):
        return "<AnalysisResult: %s>" % self.key

    def get_image(self, save_kwargs=None):
        return encode_image(self.visualized, save_kwargs=save_kwargs)


class AnalysisResultSet(object):
    def __init__(self, results):
        self.results = results

    def __repr__(self):
        return repr(self.results)

    def __getattr__(self, k):
        for result in self.results:
            if result.key == k:
                return result

    def __getitem__(self, k):
        return self.results[k]

    def __len__(self):
        return len(self.results)


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
