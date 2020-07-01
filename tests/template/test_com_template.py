from libertem.web.notebook_generator.code_template import CodeTemplate

com_default = '''
com_analysis = ctx.create_com_analysis(dataset=ds, cx=125, cy=125, mask_radius=62.5)
roi = com_analysis.get_roi()
udf = com_analysis.get_udf()
com_result = ctx.run_udf(ds, udf, roi, progress=True)
com_result = com_analysis.get_udf_results(com_result, roi)
print(com_result)
'''
com_plot = '''
fig, axes = plt.subplots()
axes.set_title("field")
axes.imshow(com_result.field.visualized)
fig, axes = plt.subplots()
axes.set_title("magnitude")
axes.imshow(com_result.magnitude.visualized)
fig, axes = plt.subplots()
axes.set_title("curl")
axes.imshow(com_result.curl.visualized)
'''


def test_com_default():
    conn = {'connection': {'type': 'local'}}
    dataset = {'type': 'HDF5', 'params': {}}
    params = {'shape': 'com', 'cx': 125, 'cy': 125, 'r': 62.5}
    comp_analysis = [{'analysisType': 'CENTER_OF_MASS', 'parameters': params}]
    instance = CodeTemplate(conn, dataset, comp_analysis)
    docs, analysis, plot = instance.analysis()[0]
    assert com_default.strip('\n') == analysis
    assert com_plot.strip('\n') == plot
