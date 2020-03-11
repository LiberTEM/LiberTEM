.. _authorship:

Authorship policy
=================

Since the performance of scientists is evaluated based on the number and quality
of their publications, authorship is an important topic for software with a
focus on scientific application, such as LiberTEM. Our goal is to give
appropriate credit to each contribution so that the work of contributors becomes
visible as a form of scientific output and is considered in the evaluation of
their productivity as scientists. We hope that this can support and encourage
Open Source software development as a form of scientific work.

Following established practices for scientific papers, we distinguish between
creators, who are persons that contributed in a way that establishes
co-authorship of LiberTEM, and contributors, who are persons that contributed in
other ways, for example through discussions or help.

Creators are persons who fulfill at least one of these criteria:

* Contributed at least one commit to the repository.
* Contributed material such as code or documentation in other ways that don't
  appear directly as commit, including prototype code that served as a basis for
  code in LiberTEM.
* Contributed to the management and direction, for example
  active management and support for contributors who dedicate a significant part
  of their working time to LiberTEM.

We maintain two files, ``packaging/creators.json`` and
``packaging/contributors.json``, where we work to keep track of creators and
contributors. For transparency, we include a short statement of the nature of
the contribution. If you feel like you or someone else should be listed there or
should be represented differently, please file a pull request or contact us!
Accidental discrepancies happen, and many eyes and helping hands help us to keep
the information up to date.

Please note that the GitHub breakdown of contributions is not always reflecting
reality. As an example, commits are only linked to GitHub accounts if the Git
client is configured correctly for that.

All people listed in ``creators.json`` are included as authors in publications
of the software itself, for example `uploads on Zenodo
<https://doi.org/10.5281/zenodo.1477847>`_.

Publications about LiberTEM
---------------------------

For other publications about LiberTEM, such as scientific papers, all
persons with at least one contribution within the past 18 months that qualifies
them for being a creator of LiberTEM will be contacted during the drafting
process of the publication, for example through their GitHub handle or via
e-mail, and offered co-authorship. If they actively consent to being a co-author
in a timely manner, they are included as co-authors under the rules of that
particular publication and medium. For scientific papers, that typically
includes an obligation to approve drafts before submission and being accountable
for all aspects of the work. Co-authors who are not responsive may be excluded
if they don't respond to a reminder, for example to approve a draft, in a timely
fashion.

For publications that only cover specific aspects of LiberTEM, for example a
particular feature, only contributions to those particular aspects are
considered to select creators who will be offered co-authorship.

Co-authors who are not creators of LiberTEM can be included in publications
about LiberTEM following established practices for authorship, for example as
contributors to the content of the publication or to other presented material,
such as scientific applications.

Ordering of author lists
------------------------

The creators and contributors are listed alphabetically in
:ref:`acknowledgments` with a short statement about their contribution.

Alphabetical ordering of author lists is uncommon for scientific papers.
Instead, the position on the author list is used to indicate the relative amount
of contribution of an author. The author that contributed most of the content is
listed first and the author that contributed most guidance is listed last.
Casual readers, in particular encountering an abbreviated author list in a
reference within a citing paper, will assign most credit to the first author. An
alphabetical author list would therefore be unfair towards main contributors
with a name in the middle of the alphabet.

In order to resolve this issue and assign prestigious author positions to the
people who deserve them in a transparent fashion, the authors agree among each
other about their authorship positions for each publication individually.

Since Zenodo assigns DOIs and allows to export a citation for reference managers
that are commonly used in scientific publishing, a Zenodo upload is treated as a
scientific publication and the authors are included in the order of
``packaging/creators.json``. This allows to assign prestigious first author
positions to main contributors when LiberTEM is cited in scientific papers.

Authorship questions should be resolved through discussion in Issues and change
proposals in the form of Pull Requests. Our goal is amicable cooperation and
proper credit for all contributions.

If you have questions or would like to suggest changes to this policy, please
contact us! See :pr:`460` for the initial discussion that lead to establishing
this policy.
