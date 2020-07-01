from libertem.web.notebook_generator.code_template import CodeTemplate

fem_default = '''
fem_analysis = FEMAnalysis(dataset=ds, parameters={'shape': 'ring', 'cx': 125, \
'cy': 125, 'ri': 62.5, 'ro': 125})
fem_result = ctx.run(fem_analysis, progress=True)
'''

fem_plot = '''
plt.figure()
plt.imshow(fem_result.intensity.visualized)
'''


def test_fem_analysis():
    conn = {'connection': {'type': 'local'}}
    dataset = {'type': 'HDF5', 'params': {}}
    params = {'shape': 'ring', 'cx': 125, 'cy': 125, 'ri': 62.5, 'ro': 125}
    comp_analysis = [{'analysisType': 'FEM', 'parameters': params}]
    instance = CodeTemplate(conn, dataset, comp_analysis)
    docs, analysis, plot = instance.analysis()[0]
    assert fem_default.strip('\n') == analysis
    assert fem_plot.strip('\n') == plot

    dependency = instance.dependency()
    temp_dep = "from libertem.analysis import FEMAnalysis"
    assert temp_dep in dependency
