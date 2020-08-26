from .code_template import CodeTemplate
from textwrap import indent


def script_generator(conn, dataset, comp, save=False):
    instance = CodeTemplate(conn, dataset, comp, type='script')
    initial_part = []
    initial_part.append(instance.dependency())
    initial_part.append(instance.initial_setup())
    initial_part.append(instance.connection())
    initial_part = '\n\n'.join(initial_part)
    second_part = []
    second_part.append(instance.dataset())

    for docs, analysis, plots, store in instance.analysis():
        second_part.append(analysis)
        for plot in plots:
            second_part.append(plot)
        if save:
            second_part.append(store)

    second_part = '\n\n'.join(second_part)
    second_part = indent(second_part, " "*8)
    return '\n\n'.join([initial_part, second_part])
