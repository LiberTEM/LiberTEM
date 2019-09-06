try:
    import locale
    locale.setlocale(locale.LC_ALL, '')
except Exception:
    pass

import click
import html5lib
from docutils.core import publish_string
from docutils.parsers.rst import roles
import xml


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

    def walk(elem, drop_first_p):
        # Convert <h1></h1> to <p><strong></strong></p>
        if elem.tag == 'h1':
            text = elem.text
            elem.clear()
            elem.tag = 'p'
            strong = xml.etree.ElementTree.Element('strong')
            strong.text = text
            elem.append(strong)
        # recurse
        for child in elem:
            # The first <p> contains the badges which don't work well on zenodo
            # FIXME removing the first <p> in the document is fragile, but for now the easiest path.
            # Do a proper solution if this stops working.
            if drop_first_p and child.tag == 'p':
                elem.remove(child)
                drop_first_p = False
            walk(child, drop_first_p)

    walk(elem, True)
    res = html5lib.serialize(elem)
    return res


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
