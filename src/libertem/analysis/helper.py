# duplicate : Maintaining this version only

from string import Template
from pypandoc import convert_text


class GeneratorHelper:

    short_name = None
    api = None
    temp_analysis = ["${short}_analysis = ctx.$analysis_api($params)",
                     "$roi",
                     "udf = ${short}_analysis.get_udf()",
                     "${short}_result = ctx.run_udf(ds, udf, roi, progress=True)"]

    def __init__(self, params):
        self.params = params

    # common in here and code_template
    def format_template(self, template, data):
        template = "\n".join(template)
        return Template(template).substitute(data)

    def get_dependency(self):
        return None

    def convert_params(self):
        return None

    def get_plot(self):
        return None

    def get_docs(self):
        return None

    def format_docs(self, docs_rst):
        """
        function to convert RST to MD format
        """
        output = convert_text(docs_rst, 'commonmark', format='rst')
        # converting heading level
        output = output.replace('#', '###')
        return output

    def get_roi_code(self):

        if 'roi' in self.params.keys():
            temp_roi = ["roi_params = $roi_params",
                        "roi = get_roi(roi_params, ds.shape.nav)"]
            data = {'roi_params': self.params['roi']}
            roi = self.format_template(temp_roi, data)
        else:
            roi = f"roi = {self.short_name}_analysis.get_roi()"

        return roi

    def get_analysis(self):

        params_ = self.convert_params()
        roi = self.get_roi_code()

        data = {'short': self.short_name,
                'analysis_api': self.api,
                'params': params_,
                'roi': roi}

        analy_ = self.format_template(self.temp_analysis, data)

        return analy_


# can be moved to some other place or any other idea!
temp_cluster_controller = """
stddev_udf = StdDevUDF()
roi = cluster_analysis.get_sd_roi()
sd_udf_results = ctx.run_udf(dataset=ds, udf=stddev_udf, roi=roi, progress=True)
sd_udf_results = consolidate_result(sd_udf_results)


center = (parameters["cy"], parameters["cx"])
rad_in = parameters["ri"]
rad_out = parameters["ro"]
n_peaks = parameters["n_peaks"]
min_dist = parameters["min_dist"]
sstd = sd_udf_results['std']
sshape = sstd.shape
if not (center is None or rad_in is None or rad_out is None):
    mask_out = 1*_make_circular_mask(center[1], center[0], sshape[1], sshape[0], rad_out)
    mask_in = 1*_make_circular_mask(center[1], center[0], sshape[1], sshape[0], rad_in)
    mask = mask_out - mask_in
    masked_sstd = sstd*mask
else:
    masked_sstd = sstd

coordinates = peak_local_max(masked_sstd, num_peaks=n_peaks, min_distance=min_dist)

y = coordinates[..., 0]
x = coordinates[..., 1]
z = range(len(y))

mask = sparse.COO(
    shape=(len(y), ) + tuple(ds.shape.sig),
    coords=(z, y, x), data=1
)

udf = ApplyMasksUDF(
    mask_factories=lambda: mask, mask_count=len(y), mask_dtype=np.uint8,
    use_sparse=True
)

udf_results = ctx.run_udf(dataset=ds, udf=udf, progress=True)
cluster_result = cluster_analysis.get_udf_results(udf_results=udf_results, roi=roi)
"""
