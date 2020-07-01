from libertem.web.notebook_generator.code_template import CodeTemplate

point_default = '''
point_analysis = ctx.create_point_analysis(dataset=ds, x=125, y=125)
roi = point_analysis.get_roi()
udf = point_analysis.get_udf()
point_result = ctx.run_udf(ds, udf, roi, progress=True)
'''
point_plot = '''
plt.figure()
plt.imshow(np.squeeze(point_result['intensity'].data))
'''


def test_point_default():
    conn = {'connection': {'type': 'local'}}
    dataset = {'type': 'HDF5', 'params': {}}
    params = {'shape': 'point', 'cx': 125, 'cy': 125}
    comp_analysis = [{'analysisType': 'APPLY_POINT_SELECTOR', 'parameters': params}]
    instance = CodeTemplate(conn, dataset, comp_analysis)
    docs, analysis, plot = instance.analysis()[0]
    assert point_default.strip('\n') == analysis
    assert point_plot.strip('\n') == plot
