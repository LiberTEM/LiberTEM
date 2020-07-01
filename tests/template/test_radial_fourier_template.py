from libertem.web.notebook_generator.code_template import CodeTemplate

radial_fourier_default = '''
radial_analysis = ctx.create_radial_fourier_analysis(dataset=ds, cx=125, cy=125, \
ri=62.5, ro=125, n_bins=1, max_order=8)
roi = radial_analysis.get_roi()
udf = radial_analysis.get_udf()
radial_result = ctx.run_udf(ds, udf, roi, progress=True)
radial_result = radial_analysis.get_udf_results(radial_result, roi)
print(radial_result)
'''

radial_fourier_plot = '''
fig, axes = plt.subplots()
axes.set_title("dominant_0")
axes.imshow(radial_result.dominant_0.visualized)
fig, axes = plt.subplots()
axes.set_title("absolute_0_0")
axes.imshow(radial_result.absolute_0_0.visualized)
fig, axes = plt.subplots()
axes.set_title("absolute_0_1")
axes.imshow(radial_result.absolute_0_1.visualized)
'''


def test_radial_fourier_default():
    conn = {'connection': {'type': 'local'}}
    dataset = {'type': 'HDF5', 'params': {}}
    params = {'shape': 'radial_fourier', 'cx': 125, 'cy': 125, 'ri': 62.5,
              'ro': 125, 'n_bins': 1, 'max_order': 8}
    comp_analysis = [{'analysisType': 'RADIAL_FOURIER', 'parameters': params}]
    instance = CodeTemplate(conn, dataset, comp_analysis)
    docs, analysis, plot = instance.analysis()[0]
    assert radial_fourier_default.strip('\n') == analysis
    assert radial_fourier_plot.strip('\n') == plot
