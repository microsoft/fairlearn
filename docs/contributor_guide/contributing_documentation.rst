.. _contributing_documentation:

Contributing documentation
--------------------------

Documentation is formatted in restructured text (ReST) and the website is
built using `Sphinx <https://www.sphinx-doc.org/en/master/>`_ and some of its
extensions.

Specifically, the website is available for most previous releases to
allow users to check the documentation of the version of the package that they are using.

Documentation structure
^^^^^^^^^^^^^^^^^^^^^^^

Most of the pages on the website are generated based on the :code:`.rst` files that can be found
in the `docs <https://github.com/fairlearn/fairlearn/tree/main/docs>`_ folder in the
repository. The documentation is generated following paths of that folder.

The home page is static and can be found in a
`subfolder <https://github.com/fairlearn/fairlearn/tree/main/docs/static_landing_page>`_.

The `API reference <file:///Users/tamara/FOSS/fairlearn/docs/_build/html/api_reference/index.html>`_
documentation is also generated from the page structure defined
in the `docs folder <https://github.com/fairlearn/fairlearn/tree/main/docs/api_reference>`_.
However, the content of the generated pages is taken from the NumPy style docstrings
found in the files of the respective classes and methods. For example, take a look at the
`code and docstrings <https://github.com/fairlearn/fairlearn/blob/main/fairlearn/preprocessing/_correlation_remover.py>`_
of the :code:`CorrelationRemover` and the corresponding
`generated page <file:///Users/tamara/FOSS/fairlearn/docs/_build/html/api_reference/generated/fairlearn.preprocessing.CorrelationRemover.html>`_.

Contributing
^^^^^^^^^^^^

The instructions below are an addition to the instructions on the
:ref:`development_process` page. Please follow the installation
instructions on that page, and return to follow the further
documentation specific guidelines.

You may also need to `install pandoc <https://pandoc.org/installing.html>`_. and :code:`matplotlib`.

.. prompt:: bash

    pip install matplotlib>=3.2.1

You can contribute updates to existing documentation by navigating to the
relevant part of the repository (typically in the `docs` directory), and
editing the restructured text files (`.rst`) corresponding to your updates.

To build the webpage run the following command from the repository root
directory:

.. prompt:: bash

    python -m sphinx -v -b html -n -j auto docs docs/_build/html

or use the shortcut:

.. prompt:: bash

    make doc

This will generate the website in the directory mentioned at the end of the
command. Rerunning this after making changes to individual files only
rebuilds the changed pages, so the build time should be a lot shorter.

You can check that the document(s) render properly by inspecting the HTML with
the following commands:

.. prompt:: bash

    #replace start with open for MacOS and xdg-open for Linux
    start docs/_build/html/index.html
    start docs/_build/html/quickstart.html
    ...
    start docs/_build/html/auto_examples/plot_*.html

The rendered HTML files can be explored in any file explorer and then opened
using a browser (e.g., Chrome/Firefox/Safari).

:code:`plot_*` can be replaced with any of the notebooks in the
:code:`auto_examples` folder. To view your changes, simply navigate to the
relevant part of the website and check that your updates render properly
and links work as expected.

Note that some changes to documentation may involve modifying several files
(i.e: index files, other documents in which the current one should be linked).
Be sure to modify all of the relevant documents and use the commands above to
ensure that they all render properly.

.. note::

    If you encounter problems with the documentation build (locally or in your
    pull request) and need help, simply @mention
    :code:`@fairlearn/fairlearn-maintainers` to get help.


Citations
^^^^^^^^^

Citations are built using the `sphinxcontrib-bibtex <https://pypi.org/project/sphinxcontrib-bibtex/>`_
extension. This allows us to use the `refs.bib <https://github.com/fairlearn/fairlearn/blob/main/docs/refs.bib>`_
BibTeX file to generate our citations.

To add a citation:

1. Check if your required BibTeX entry already exists in the
   `docs/refs.bib <https://github.com/fairlearn/fairlearn/blob/main/docs/refs.bib>`_ file.
   If not, simply paste your entry at the end.
2. Change your bibtex id to the format ``<author-last-name><4digit-year><keyword>``.
3. Use the :code:`:footcite:`bibtex-id`` role to create an inline citation rendered as :code:`[CitationNumber]`.
   For example, :code:`:footcite:`agarwal2018reductions`` will be rendered as :footcite:`agarwal2018reductions`.
4. You can also use :code:`:footcite:t:`bibtex-id`` to create a textual citation.
   The role :code:`:footcite:t:`agarwal2018reductions`` will be rendered as :footcite:t:`agarwal2018reductions`.
5. To add the bibliography use :code:`.. footbibliography::` directive at the bottom of your file if not already present.
   This will list all the citations for the current document.

   For example :code:`.. footbibliography::` will be rendered as shown below:

   .. footbibliography::
