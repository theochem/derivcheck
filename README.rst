.. image:: https://travis-ci.org/tovrstra/derivcheck.svg?branch=master
    :target: https://travis-ci.org/tovrstra/derivcheck
.. image:: https://anaconda.org/tovrstra/derivcheck/badges/version.svg
    :target: https://anaconda.org/tovrstra/derivcheck

Derivcheck provides a robust and very sensitive checker of analytic partial
derivates. It is intended to be used in unit tests of other projects. See
``deriv_check/basic_example.py`` for a basic example.


Installation
============

Derivcheck can be installed with pip (system wide or in a virtual environment):

.. code:: bash

    pip install derivcheck

Alternatively, you can install derivcheck in your home directory:

.. code:: bash

    pip install derivcheck --user

Lastly, you can also install derivcheck with conda:

.. code:: bash

    conda install -c tovrstra derivcheck


Background and usage
====================

This module implements a function ``assert_deriv``, which uses Ridders' numerical finite
difference scheme to test the implementation of analytic finite differences. Ridders'
method automatically finds the step size (given an initial upper estimate) and the
polynomial order that result in the best approximation. In practice, this means that 14
digits of precision can be reached with 6 to 12 evaluations of the function of interest.

The implementation of Ridders' method is based on the one from the book "Numerical
Recipes" (http://numerical.recipes/), which is in turn a slight rendition of the original
method as proposed by Ridders. (Ridders, C.J.F. 1982, Advances in Engineering Software,
vol. 4, no. 2, pp. 75–76. https://doi.org/10.1016/S0141-1195(82)80057-0)

It is assumed that you have implemented two functions: ``f`` and its derivative or
gradient ``g``. The function ``f`` takes one argument: a scalar or array with shape
``shape_in``. It returns a scalar or an array with shape ``shape_out``. The function ``g``
has the same input but returns a scalar or an array with shape (``shape_out + shape_in``).

The consistency of ``f`` and ``g``, can then be tested around a input value ``origin``
with the following code:

.. code:: python

    assert_deriv(f, g, origin)

where ``origin`` is a scalar or array with shape ``shape_in``, depending on what ``f`` and
``g`` expect as input. An ``AssertionError`` is raised when the gradient function ``g`` is
not consistent with numerical derivatives of ``f``. If Ridders' method does not converge
to sufficiently accurate estimates of a derivative, a ``FloatingPointError`` is raised.

The function ``assert_deriv`` takes several optional arguments to tune its behavior:


* ``widths`` : ``float`` or ``np.ndarray`` (default ``1e-4``)

  The initial (maximal) step size for the finite difference method. Do not take a value
  that is too small. When an array is given, each matrix element of the input of the
  function gets a different step size. When a matrix element is set to zero, the
  derivative towards that element is not tested. The function will not be sampled beyond
  [origin-widths, origin+widths].

* ``output_mask`` : ``np.ndarray`` or ``None`` (default)

  This option is useful when the function returns an array output: it allows the caller to
  select which components of the output need to be tested. When not given, all components
  are tested.

* ``rtol`` : ``float``

  The allowed relative error on the derivative.

* ``atol`` : ``float``

  The allowed absolute error on the derivative.


Release history
===============

- **2017-07-30** 1.0.2

  Updated README and install recipe for Conda

- **2017-07-30** 1.0.1

  Fix some missing files and extend README

- **2017-07-28** 1.0.0

  - Ridders' finite difference scheme for testing analytic derivatives.
  - Fully deterministic procedure.
  - More intuitive API

- **2017-07-27** 0.1.0

  Code is made Python 3 compatible and still works with 2.7. Some packaging
  improvements.

- **2017-07-27** 0.0.0

  Initial version: code taken from the Romin project (with contributions and
  ideas from Michael Richer and Paul W. Ayers). Some bugs were fixed through QA
  and CI (pylint, pycodestyle, pydocstyle, nosetests and coverage).


Testing
=======

First you need to get the source package and unzip it, or you can clone the repository. In
the source tree, you simply run:

.. code:: bash

    ./setup.py nosetests


How to make a release (Github, PyPI and anaconda.org)
=====================================================

Before you do this, make sure everything is OK. The PyPI steps cannot be undone. If you
delete a file from PyPI (because of a mistake), you cannot upload the fixed file with the
same filename! See https://github.com/pypa/packaging-problems/issues/74

The following steps are tested on an Linux system, with Miniconda and twine installed. In
your conda environment, you also need to install ``conda-build`` and ``anaconda-client``.

1. Update the ``__version__`` variable in ``derivhceck.py`` if not done yet. Make use of
   semantic versioning: http://semver.org/
2. Update the release history.
3. Commit the final changes to master and push to github.
4. Wait for the tests to pass. Check if the website looks ok, etc. If not, fix things and
   repeat step 3.
5. Make a git version tag: ``git tag $(python -c 'import derivcheck; print derivcheck.__version__')``
6. Push to github with tags: ``git push origin master --tags``
7. Make a source archive: ``./setup.py sdist``
8. Upload the source tar file to github.com, using your browser. See
   https://help.github.com/articles/creating-releases/
9. Upload the source tar file to PyPI: ``twine upload dist/derivhceck*.tar.gz``
10. Get the sha256 checksum of the source file: ``sha256sum dist/derivcheck*.tar.gz``
11. Update the ``version`` and ``sha256`` fields in ``conda/meta.yml``.
12. Build the conda package: ``conda build conda/`` Take note of the location of the
    package for the following step.
13. Upload the conda package: ``anaconda login; anaconda upload <package path>``
14. Commit the updated conda file and push to github.

This is not ideal yet because the changes in the conda file are committed after the
release. Idealy the conda file gets hosted on https://conda-forge.org/.

In future, this should become fully automated: as soon as a tag is pushed with a version
number, the entire process should be carried out automatically. A few special things are
needed to make this work:

- Include all of the above steps in the Travis script. A release should only be made if
  all tests pass.

  - General Travis deployment docs: https://docs.travis-ci.com/user/deployment/
  - Documentation for Github releases: https://docs.travis-ci.com/user/deployment/releases/
  - Documentation for Pypi releases: https://docs.travis-ci.com/user/deployment/pypi/
  - Example for anaconda: https://gist.github.com/yoavram/05a3c04ddcf317a517d5

- Some more jinja tricks are needed in the meta.yml files, which we have to render
  before passing to `conda build`, to fill in version and sha256 sum.
- Anaconda, Pypi and Github credentials should somehow be known to Travis. To do this
  safely, encryption is needed, which is explained here:
  https://docs.travis-ci.com/user/encryption-keys/
- Anaconda tokens are ideal for accessing the repo with limited features:
  https://docs.continuum.io/anaconda-cloud/user-guide/tasks/work-with-accounts#creating-access-tokens
- A distinction should be made between alpha, beta and stable releases:

  - PyPI does not allow separate "channels" for alpha and beta releases. Only stable
    releases should be uploaded. If not, people will just upgrade into development
    versions without realizing it.
  - Anaconda labels can be used to mark alpha and beta releases, default is stable (main).
  - Github can make a distinction between stable and pre- releases.
