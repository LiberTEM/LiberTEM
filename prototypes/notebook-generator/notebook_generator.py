import nbformat as nbf
from code_template import CodeTemplate


class notebook:

    def __init__(self):
        self.nb = nbf.v4.new_notebook()

    def add_code(self, code_):
        new_cell = nbf.v4.new_code_cell(code_)
        self.nb['cells'].append(new_cell)

    def add_doc(self, doc_):
        new_cell = nbf.v4.new_markdown_cell(doc_)
        self.nb['cells'].append(new_cell)

    def generate(self):
        fname = 'comp_analysis_notebook_1.ipynb'
        with open(fname, 'w') as f:
            nbf.write(self.nb, f)


def notebook_generator(conn, dataset, comp):
    # initialization
    nb = notebook()
    instance = CodeTemplate(conn, dataset, comp)

    nb.add_code(instance.dependency())
    nb.add_code(instance.initial_setup())

    ctx, conn_docs = instance.connection()
    nb.add_doc(conn_docs)
    nb.add_code(ctx)
    nb.add_code(instance.dataset())

    for docs, analysis, plot in instance.analysis():
        nb.add_doc(docs)
        nb.add_code(analysis)
        nb.add_code(plot)

    nb.generate()


if __name__ == '__main__':
    # ring_details = {'analysisType': 'APPLY_RING_MASK', 'parameters': {'shape': 'ring', 'cx': 125, 'cy': 125, 'ri': 62.5, 'ro': 125}}
    sum_details = {'analysisType': 'SUM_FRAMES', 'parameters': {'roi': {'shape': 'disk', 'cx': 42, 'cy': 50, 'r': 10.5}}}
    # pick_details = {'analysisType': "PICK_FRAME", 'parameters': {'x': 21, 'y': 59}}
    # com_details = {'analysisType': 'CENTER_OF_MASS', 'parameters': {'shape': 'com', 'cx': 125, 'cy': 125, 'r': 57.99711815561959}}
    radial_details = {'analysisType': 'RADIAL_FOURIER', 'parameters': {'shape': 'radial_fourier', 'cx': 125, 'cy': 125, 'ri': 62.5, 'ro': 125, 'n_bins': 1, 'max_order': 8}}
    comp_an = [sum_details, radial_details]

    conn = {'type': 'local', 'url': 'http://tcp.localhost:9000'}

    dataset = {"type": "HDF5", "params": {'path': "/home/abi/Documents/LiberTEM_data/calibrationData_circularProbe.h5",
               'ds_path': "4DSTEM_experiment/data/datacubes/polyAu_4DSTEM/data"}}

    notebook_generator(conn, dataset, comp_an)
