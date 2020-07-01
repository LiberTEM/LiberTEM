from libertem.web.notebook_generator.code_template import CodeTemplate

temp_dep = '''
import matplotlib.pyplot as plt
import libertem.api as lt
import numpy as np
from libertem.analysis.getroi import get_roi
import numpy as np
'''

temp_ds = '''
params = {'path': '/home/path/name.h5', 'dspath': '/name/dspath'}
ds = ctx.load("HDF5", **params)
'''

temp_cluster = '''
cluster = executor.dask.DaskJobExecutor.connect("tcp://url")
ctx = lt.Context(executor=cluster)
'''


def test_dependency():
    conn = {'connection': {'type': 'local'}}
    dataset = {'type': 'HDF5', 'params': {}}
    comp_analysis = [{'analysisType': 'SUM_FRAMES', 'parameters': {'roi': {}}}]
    instance = CodeTemplate(conn, dataset, comp_analysis)
    dep = instance.dependency()
    assert temp_dep.strip('\n') == dep


def test_conn_local():
    conn = {'connection': {'type': 'local'}}
    dataset = {'type': 'HDF5', 'params': {}}
    comp_analysis = [{'analysisType': 'SUM_FRAMES', 'parameters': {'roi': {}}}]
    instance = CodeTemplate(conn, dataset, comp_analysis)
    ctx, _ = instance.connection()
    assert ctx == "ctx = lt.Context()"


def test_conn_cluster():
    conn = {'connection': {'type': 'cluster', 'url': 'tcp://url'}}
    dataset = {'type': 'HDF5', 'params': {}}
    comp_analysis = [{'analysisType': 'SUM_FRAMES', 'parameters': {'roi': {}}}]
    instance = CodeTemplate(conn, dataset, comp_analysis)
    ctx, _ = instance.connection()
    assert temp_cluster.strip('\n') == ctx


def test_dataset():
    conn = {'connection': {'type': 'local'}}
    dataset = {'type': 'HDF5', 'params': {'path': '/home/path/name.h5', 'dspath': '/name/dspath'}}
    comp_analysis = [{'analysisType': 'SUM_FRAMES', 'parameters': {'roi': {}}}]
    instance = CodeTemplate(conn, dataset, comp_analysis)
    ds = instance.dataset()
    assert temp_ds.strip('\n') == ds
