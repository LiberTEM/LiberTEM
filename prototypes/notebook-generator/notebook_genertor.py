import nbformat as nbf
from code_template import CodeTemplate


def notebook_generator(conn, dataset, comp):
    # initialization
    nb = nbf.v4.new_notebook()
    instance = CodeTemplate(conn, dataset, comp)

    nb['cells'].append(nbf.v4.new_code_cell(instance.dependency()))
    nb['cells'].append(nbf.v4.new_code_cell(instance.modifiers()))
    ctx, conn_docs = instance.connection()
    nb['cells'].append(nbf.v4.new_markdown_cell(conn_docs))
    nb['cells'].append(nbf.v4.new_code_cell(ctx))
    nb['cells'].append(nbf.v4.new_code_cell(instance.dataset()))
    for docs, analysis, plot in instance.analysis():
        nb['cells'].append(nbf.v4.new_markdown_cell(docs))
        nb['cells'].append(nbf.v4.new_code_cell(analysis))
        nb['cells'].append(nbf.v4.new_code_cell(plot))

    fname = 'analysis_notebook.ipynb'
    with open(fname, 'w') as f:
        nbf.write(nb, f)


if __name__ == '__main__':
    comp_an = {"SumAnalysis": {'roi': ''},
               "RingMaskAnalysis": {'shape': 'ring', 'cx': 125, 'cy': 125, 'ri': 62.5, 'ro': 125}}

    conn = {'type': 'local', 'url': 'http://tcp.localhost:9000'}

    dataset = {"type":"HDF5", "params": {'path': "/home/abi/Documents/LiberTEM_data/calibrationData_circularProbe.h5",
               'ds_path': "4DSTEM_experiment/data/datacubes/polyAu_4DSTEM/data"}}

    notebook_generator(conn, dataset, comp_an)
