from libertem.web.notebook_generator.code_template import CodeTemplate

ring_default = '''
ring_analysis = ctx.create_ring_analysis(dataset=ds, cx=125, cy=125, ri=62.5, ro=125)
roi = ring_analysis.get_roi()
udf = ring_analysis.get_udf()
ring_result = ctx.run_udf(ds, udf, roi, progress=True)
'''

ring_plot = '''
plt.figure()
plt.imshow(np.squeeze(ring_result['intensity'].data))
'''


def test_ring_default():
    conn = {'connection': {'type': 'local'}}
    dataset = {'type': 'HDF5', 'params': {}}
    params = {'shape': 'ring', 'cx': 125, 'cy': 125, 'ri': 62.5, 'ro': 125}
    comp_analysis = [{'analysisType': 'APPLY_RING_MASK', 'parameters': params}]
    instance = CodeTemplate(conn, dataset, comp_analysis)
    docs, analysis, plot = instance.analysis()[0]
    assert ring_default.strip('\n') == analysis
    assert ring_plot.strip('\n') == plot
