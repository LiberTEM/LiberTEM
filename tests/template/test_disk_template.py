from libertem.web.notebook_generator.code_template import CodeTemplate

disk_default = '''
disk_analysis = ctx.create_disk_analysis(dataset=ds, cx=125, cy=125, r=62.5)
roi = disk_analysis.get_roi()
udf = disk_analysis.get_udf()
disk_result = ctx.run_udf(ds, udf, roi, progress=True)
'''
disk_plot = '''
plt.figure()
plt.imshow(np.squeeze(disk_result['intensity'].data))
'''


def test_disk_default():
    conn = {'connection': {'type': 'local'}}
    dataset = {'type': 'HDF5', 'params': {}}
    params = {'shape': 'disk', 'cx': 125, 'cy': 125, 'r': 62.5}
    comp_analysis = [{'analysisType': 'APPLY_DISK_MASK', 'parameters': params}]
    instance = CodeTemplate(conn, dataset, comp_analysis)
    docs, analysis, plot = instance.analysis()[0]
    assert disk_default.strip('\n') == analysis
    assert disk_plot.strip('\n') == plot
