.. _development_process:

Development process
-------------------

Development happens against the :code:`main` branch following the
`GitHub flow model <https://guides.github.com/introduction/flow/>`_.
Contributors should use their own forks of the repository. In their fork, they
create feature branches off of :code:`main`, and their pull requests should
target the :code:`main` branch. Maintainers are responsible for prompt
review of pull requests.

Pull requests against :code:`main` trigger automated tests that are run
through Azure DevOps, GitHub Actions, and CircleCI. Additional test suites are
run periodically. When adding new code paths or features, tests are a
requirement to complete a pull request. They should be added in the
:code:`test` directory.

Documentation should be provided with pull requests that add or change
functionality. This includes comments in the code itself, docstrings, and user
guides. For exceptions to this rule the pull request author should coordinate
with a maintainer. For changes that fix bugs, add new features, change APIs,
etc., i.e., for changes that are relevant to developers and/or users please
also add an entry in :ref:`version_guide` in the section corresponding to the
*next* release, since that's where your change will be included.
If you're a new contributor please also add yourself to
`AUTHORS.md <https://github.com/fairlearn/fairlearn/blob/main/AUTHORS.md>`_.

Docstrings should follow
`numpydoc format <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
This is a `recent decision by the community <https://github.com/fairlearn/fairlearn/issues/314>`_.
The new policy is to update docstrings that a PR touches, as opposed to
changing all the docstrings in one PR.


Installation instructions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You will need a GitHub account to contribute to this project. Authenticating with Github
using an ssh connection is highly recommended. You can learn more about it
`here <https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account>`_.

First, clone the repository locally via:

.. prompt:: bash

   git clone git@github.com:fairlearn/fairlearn.git


(Optional) Set up a virtual environment:

   While you can use :code:`pip` to install the fairlearn package globally, we strongly
   recommend developing using a virtual environment. Virtual environments are a great way
   to isolate project dependencies, especially if you're working on multiple issues that
   require different versions of the package.

   To create a Python virtual environment using :code:`conda`, make your virtual environment
   inside the fairlearn project directory. This will create a new :code:`fairlearn` folder
   and seed it with a :code:`python=3.12` environment. Then activate the virtual environment:

   .. prompt:: bash

      conda create --name fairlearn python=3.12  # creates virtual env
      conda activate fairlearn # activates virtual env
      (fairlearn) $ # notice the shell prompt includes name of active virtual env

   If you see the name of the environment in the shell prompt,
   then the virtual environment is activated. To deactivate it, run: :code:`conda deactivate`.

To install in editable mode, from the repository root path run:

.. prompt:: bash

   pip install -e .

To verify that the code works as expected run:

.. prompt:: bash

   python ./scripts/install_requirements.py --pinned False
   python -m pytest -s ./test/unit

.. note::

   If there is a :code:`torch` related error during the installation,
   please downgrade keep your :code:`python` version between 3.9-3.12.

Fairlearn currently includes plotting functionality provided by
:code:`matplotlib`. This is for a niche use case, so it isn't a default requirement. To install run:

.. prompt:: bash

   pip install -e .
   pip install matplotlib

The Requirements Files
""""""""""""""""""""""

The prerequisites for Fairlearn are split between three separate files:

* `requirements.txt <https://github.com/fairlearn/fairlearn/blob/main/requirements.txt>`_
  contains the prerequisites for the core Fairlearn package

* `requirements-dev.txt <https://github.com/fairlearn/fairlearn/blob/main/requirements-dev.txt>`_ contains
  the prerequisites for Fairlearn development (such as :code:`ruff` and :code:`pytest`)

The `requirements.txt <https://github.com/fairlearn/fairlearn/blob/main/requirements.txt>`_
file is consumed
by `setup.py <https://github.com/fairlearn/fairlearn/blob/main/setup.py>`_ to specify the dependencies to be
documented in the wheel files.
To help simplify installation of the prerequisites, we have the
`install_requirements.py <https://github.com/fairlearn/fairlearn/blob/main/scripts/install_requirements.py>`_
script which runs :code:`pip install` on both the above files.
This script will also optionally pin the requirements to any lower bound specified (by changing any
occurrences of :code:`>=` to :code:`==` in each file).

.. _contributing_pull_requests:

Contributing a pull request
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Follow the steps below to create a pull request.

#. Get a `GitHub account <https://github.com/>`_.

#. Install `GIT <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_.

#. Look at Fairlearn's issues on GitHub, specifically the ones marked
   `"help wanted" <https://github.com/fairlearn/fairlearn/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22>`_.
   Within this category we've marked issues with labels:

   * `"good first issue" <https://github.com/fairlearn/fairlearn/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22+label%3A%22good+first+issue%22>`_:
     Issues suitable for first time contributors, including people with no prior experience with coding or GitHub.
     This is an excellent way to get started!

   * `"easy" <https://github.com/fairlearn/fairlearn/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22+label%3A%22easy%22>`_:
     Issues suitable for folks with at least a bit of experience and/or able to allocate some time to look for a solution.

   *  Neither of the two above:
      issues that are demanding or awaiting scope. Likely to take more than a day or two.
      If you think this is something for you, please:

      * Identify an issue that you would like to work on.
      * Leave a comment on the issue indicating interest and outlining possible questions.
      * Once we know you are working on it, we will support you on your contribution journey!

.. note::

   If you claim an issue, please try to keep it updated each week, either by continuing a discussion in the issue itself or in a pull request.
   Issues which are not receiving updates may be claimed by someone else.

#. The communication channels are outlined here: :ref:`communication`.

#. Fork the `project repository
   <https://github.com/fairlearn/fairlearn.git>`__ by clicking on the 'Fork'
   button near the top of the page. This creates a copy of the code on your GitHub user account.
   For more details on how to fork a
   repository see `this guide <https://help.github.com/articles/fork-a-repo/>`_.

#. Clone your fork of the fairlern repo from your GitHub account to your
   local machine:

   .. prompt:: bash

      git clone git@github.com:YourLogin/fairlearn.git  # add --depth 1 if your connection is slow
      cd fairlearn

#. Add the ``upstream`` remote. This saves a reference to the main
   fairlearn repository, which you can use to keep your repository
   synchronized with the latest changes:

   .. prompt:: bash

      $ git remote add upstream git@github.com:fairlearn/fairlearn.git

#. Check that the :code:`upstream` and :code:`origin` remote aliases are configured correctly
   by running

   :code:`git remote -v` which should display:

   .. code-block:: text

        origin	git@github.com:YourLogin/fairlearn.git (fetch)
        origin	git@github.com:YourLogin/fairlearn.git (push)
        upstream	git@github.com:fairlearn/fairlearn.git (fetch)
        upstream	git@github.com:fairlearn/fairlearn.git (push)


#. (Optional) Install `pre-commit <https://pre-commit.com/#install>`_ to run code style checks before each commit:

   .. prompt:: bash

      pip install pre-commit
      pre-commit install

   Pre-commit checks can be disabled for a particular commit with :code:`git commit -n`.

#. To contribute, you will need to create a branch on your forked repository and make a pull request to the original fairlearn repository.
   Detailed description of this process you can find here:

   * `Create a branch <https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project#creating-a-branch-to-work-on>`_.
   * `Commit and push changes <https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project#making-and-pushing-changes>`_.
   * `Open a pull request <https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project#making-a-pull-request>`_.


      * Build the website following the guidelines in :ref:`contributing_documentation` and run the tests if necessary.
      * Opening a pull request comes with filling up an already provided description template.
        Please fill it up! If you created the pull request in response to an issue add :code:`#<issue-number>` for reference.

#. Celebration time! We would like to encourage you to become a part of our Fairlearn community. To do so, join our communication channels: :ref:`communication`.

Investigating automated test failures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For every pull request to :code:`main` with automated tests, you can check
the logs of the tests to find the root cause of failures. Our tests currently
run through Azure Pipelines with steps for setup, testing, and teardown. The
:code:`Checks` tab of a pull request contains a link to the
`Azure Pipelines page <dev.azure.com/responsibleai/fairlearn/_build/results>`_),
where you can review the logs by clicking on a specific step in the automated
test sequence. If you encounter problems with this workflow, please reach out
through `GitHub issues <https://github.com/fairlearn/fairlearn/issues>`_.

To run the same tests locally, find the corresponding pipeline definition (a
:code:`yml` file) in the :code:`devops` directory. It either directly contains
the command to execute the tests (usually starting with
:code:`python -m pytest`) or it refers to a template file with the command.
