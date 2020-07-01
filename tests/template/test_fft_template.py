from libertem.web.notebook_generator.code_template import CodeTemplate

sum_fft_default = '''
sumfft_analysis = SumfftAnalysis(dataset=ds, parameters={'real_rad': 62.5, \
'real_centerx': 125, 'real_centery': 125})
sumfft_result = ctx.run(sumfft_analysis, progress=True)
'''
sum_fft_plot = '''
plt.figure()
plt.imshow(sumfft_result.intensity.visualized)
'''

fft_default = '''
fft_analysis = ApplyFFTMask(dataset=ds, parameters={'rad_in': 62.5, \
'rad_out': 125, 'real_rad': 62.5, 'real_centerx': 125, 'real_centery': 125})
fft_result = ctx.run(fft_analysis, progress=True)
'''

fft_plot = '''
plt.figure()
plt.imshow(fft_result.intensity.visualized)
'''


def test_sum_fft_default():
    conn = {'connection': {'type': 'local'}}
    dataset = {'type': 'HDF5', 'params': {}}
    params = {'real_rad': 62.5, 'real_centerx': 125, 'real_centery': 125}
    comp_analysis = [{'analysisType': 'FFTSUM_FRAMES', 'parameters': params}]
    instance = CodeTemplate(conn, dataset, comp_analysis)
    docs, analysis, plot = instance.analysis()[0]
    assert sum_fft_default.strip('\n') == analysis
    assert sum_fft_plot.strip('\n') == plot

    dependency = instance.dependency()
    temp_dep = "from libertem.analysis import SumfftAnalysis"
    assert temp_dep in dependency


def test_fft_analysis():
    conn = {'connection': {'type': 'local'}}
    dataset = {'type': 'HDF5', 'params': {}}
    params = {'rad_in': 62.5, 'rad_out': 125, 'real_rad': 62.5,
              'real_centerx': 125, 'real_centery': 125}
    comp_analysis = [{'analysisType': 'APPLY_FFT_MASK', 'parameters': params}]
    instance = CodeTemplate(conn, dataset, comp_analysis)
    docs, analysis, plot = instance.analysis()[0]
    assert fft_default.strip('\n') == analysis
    assert fft_plot.strip('\n') == plot

    dependency = instance.dependency()
    temp_dep = "from libertem.analysis import ApplyFFTMask"
    assert temp_dep in dependency
