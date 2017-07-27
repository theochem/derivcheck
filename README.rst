Derivcheck provides a robust and very sensitive checker of analytic partial
derivates. It is intended to be used in unit tests of other projects. See
`basic_example.py` for a basic example.

Installation
============

Derivcheck can be installed with pip (system wide or in a virtual environment):

.. code:: bash

    pip install derivcheck

Alternatively, you can install derivcheck in your home directory:

.. code:: bash

    pip install derivcheck --user


Testing
=======

Use ``setup.py`` to run the tests:

.. code:: bash

    ./setup.py nosetests


Description of the algorithm
============================

TODO



Release history
===============

- **2017-07-27** 0.1.0

  Code is made Python 3 compatible and still works with 2.7. Some packaging
  improvements.

- **2017-07-27** 0.0.0

  Initial version: code taken from the Romin project (with contributions and
  ideas from Michael Richer and Paul W. Ayers). Some bugs were fixed through QA
  and CI (pylint, pycodestyle, pydocstyle, nosetests and coverage).
