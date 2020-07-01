from libertem.web.notebook_generator.notebook_generator import notebook_generator
import nbformat


def test_notebook_generator():
    conn = {'connection': {'type': 'local'}}
    dataset = {'type': 'HDF5', 'params': {}}
    comp_analysis = [{'analysisType': 'SUM_FRAMES', 'parameters': {'roi': {}}}]
    notebook = notebook_generator(conn, dataset, comp_analysis)
    notebook = nbformat.reads(notebook.getvalue(), as_version=4)
    assert nbformat.validate(notebook) is None
