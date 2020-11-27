from .code_template import CodeTemplate


def copy_notebook(conn, dataset, comp):
    instance = CodeTemplate(conn, dataset, comp)
    analy = []
    for analysis in instance.analysis():
        analy.append({
            'analysis': analysis['code'],
            'plot': analysis['plots']
        })

    return {
        'dependency': instance.dependency(),
        'initial_setup': instance.initial_setup(),
        'ctx': instance.connection()[0],
        'dataset': instance.dataset(),
        'analysis': analy
    }
