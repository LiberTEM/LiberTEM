from libertem.web.notebook_generator.code_template import CodeTemplate

pick_default = '''
pick_analysis = ctx.create_pick_analysis(dataset=ds ,x=42 ,y=50)
roi = pick_analysis.get_roi()
udf = pick_analysis.get_udf()
pick_result = ctx.run_udf(ds, udf, roi, progress=True)
'''

pick_plot = '''
plt.figure()
plt.imshow(np.squeeze(pick_result['intensity'].data))
'''


def test_pick_analysis():
    conn = {'connection': {'type': 'local'}}
    dataset = {'type': 'HDF5', 'params': {}}
    params = {'x': 42, 'y': 50}
    comp_analysis = [{'analysisType': 'PICK_FRAME', 'parameters': params}]
    instance = CodeTemplate(conn, dataset, comp_analysis)
    docs, analysis, plot = instance.analysis()[0]
    assert pick_default.strip('\n') == analysis
    assert pick_plot.strip('\n') == plot
