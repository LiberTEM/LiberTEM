from libertem.web.notebook_generator.code_template import CodeTemplate

sum_default = '''
sum_analysis = ctx.create_sum_analysis(dataset=ds)
roi_params = {}
roi = get_roi(roi_params, ds.shape.nav)
udf = sum_analysis.get_udf()
sum_result = ctx.run_udf(ds, udf, roi, progress=True)
'''

sum_roi = '''
sum_analysis = ctx.create_sum_analysis(dataset=ds)
roi_params = {'shape': 'disk', 'cx': 42, 'cy': 50, 'r': 10.5}
roi = get_roi(roi_params, ds.shape.nav)
udf = sum_analysis.get_udf()
sum_result = ctx.run_udf(ds, udf, roi, progress=True)
'''
sum_plot = '''
plt.figure()
plt.imshow(sum_result['intensity'].raw_data)
'''


def test_sum_default():
    conn = {'connection': {'type': 'local'}}
    dataset = {'type': 'HDF5', 'params': {}}
    comp_analysis = [{'analysisType': 'SUM_FRAMES', 'parameters': {'roi': {}}}]
    instance = CodeTemplate(conn, dataset, comp_analysis)
    docs, analysis, plot = instance.analysis()[0]
    assert sum_default.strip('\n') == analysis
    assert sum_plot.strip('\n') == plot


def test_sum_roi():
    conn = {'connection': {'type': 'local'}}
    dataset = {'type': 'HDF5', 'params': {}}
    roi_params = {'shape': 'disk', 'cx': 42, 'cy': 50, 'r': 10.5}
    comp_analysis = [{'analysisType': 'SUM_FRAMES', 'parameters': {'roi': roi_params}}]
    instance = CodeTemplate(conn, dataset, comp_analysis)
    docs, analysis, plot = instance.analysis()[0]
    assert sum_roi.strip('\n') == analysis
