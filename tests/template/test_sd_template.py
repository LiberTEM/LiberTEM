from libertem.web.notebook_generator.code_template import CodeTemplate

sd_default = '''
sd_analysis = SDAnalysis(dataset=ds, parameters={'roi': {}})
sd_result = ctx.run(sd_analysis, progress=True)
'''

sd_roi = '''
sd_analysis = SDAnalysis(dataset=ds, parameters={'roi': {'shape': 'disk', 'cx': \
42, 'cy': 50, 'r': 10.5}})
sd_result = ctx.run(sd_analysis, progress=True)
'''

sd_plot = '''
plt.figure()
plt.imshow(sd_result.intensity.visualized)
'''


def test_sd_default():
    conn = {'connection': {'type': 'local'}}
    dataset = {'type': 'HDF5', 'params': {}}
    params = {'roi': {}}
    comp_analysis = [{'analysisType': 'SD_FRAMES', 'parameters': params}]
    instance = CodeTemplate(conn, dataset, comp_analysis)
    docs, analysis, plot = instance.analysis()[0]
    assert sd_default.strip('\n') == analysis
    assert sd_plot.strip('\n') == plot

    dependency = instance.dependency()
    temp_dep = "from libertem.analysis import SDAnalysis"
    assert temp_dep in dependency


def test_sd_roi():
    conn = {'connection': {'type': 'local'}}
    dataset = {'type': 'HDF5', 'params': {}}
    params = {'roi': {'shape': 'disk', 'cx': 42, 'cy': 50, 'r': 10.5}}
    comp_analysis = [{'analysisType': 'SD_FRAMES', 'parameters': params}]
    instance = CodeTemplate(conn, dataset, comp_analysis)
    docs, analysis, plot = instance.analysis()[0]
    assert sd_roi.strip('\n') == analysis
    assert sd_plot.strip('\n') == plot

    dependency = instance.dependency()
    temp_dep = "from libertem.analysis import SDAnalysis"
    assert temp_dep in dependency
