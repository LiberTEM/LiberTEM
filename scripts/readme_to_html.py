try:
    import locale
    locale.setlocale(locale.LC_ALL, '')
except Exception:
    pass

import click
import html5lib
from docutils.core import publish_string
from docutils.parsers.rst import roles


def fake_cite(name, rawtext, text, lineno, inliner,
              options={}, content=[]):
    """
    to prevent errors with sphinxcontrib bibtex roles, we register a fake here
    """
    return [], []


roles.register_local_role("cite", fake_cite)


def extract(html_string):
    document = html5lib.parse(html_string, namespaceHTMLElements=False)
    elem = document.find(".//div[@class='document']")
    res = []
    for child in elem:
        res.append(html5lib.serialize(child))
    return "\n".join(res)


@click.command()
@click.argument('src_fname', type=str)
@click.argument('dest_fname', type=str)
def main(src_fname, dest_fname):
    with open(src_fname, "r") as srcf, open(dest_fname, "w") as destf:
        readme_html = publish_string(srcf.read(), writer_name="html")
        extracted = extract(readme_html)
        destf.write(extracted)


if __name__ == "__main__":
    main()
