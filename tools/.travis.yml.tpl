env:
  matrix:
  - MYCONDAPY=2.7
  - MYCONDAPY=3.5
  - MYCONDAPY=3.6
  global:
    - secure: ${TPL_ANACONDA_TOKEN}
    - secure: ${TPL_GITHUB_TOKEN}
    - secure: ${TPL_PYPI_PASSWORD}

# Do not use Travis Python to save some time.
language: generic
os:
  - linux
  - osx
osx_image: xcode6.4
dist: trusty
sudo: false

branches:
  only:
    - master
    - /^[0-9]+\.[0-9]+(\.[0-9]+)?([ab][0-9]+)?$/

install:
# Get miniconda. Take the right version, so re-installing python is only needed for 3.5.
- if [[ "$MYCONDAPY" == "2.7" ]]; then
    if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
      wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh;
    else
      wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh -O miniconda.sh;
    fi;
  else
    if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
      wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
    else
      wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh;
    fi;
  fi
- bash miniconda.sh -b -p $HOME/miniconda
- source $HOME/miniconda/bin/activate
- hash -r

# Configure conda and get a few essentials
- conda config --set always_yes yes
# Needed for compiler('c') function, temporary solution:
# https://github.com/conda/conda-build/issues/2263
- conda config --add channels c3i_test
- conda update -q conda
# Get the right python version for building. This only does something for 3.5.
# Install extra package needed to make things work. Most things can be listed as
# dependencies on metal.yaml and setup.py, unless setup.py already imports them.
# Install conda tools for packaging and uploading
- conda install python=${MYCONDAPY} numpy conda-build anaconda-client
- conda info -a

# Pip has more recent versions of the lint tools than plain conda.
# Conda-forge broke at some point while writing this, so I'd rather avoid it.
- pip install pydocstyle pycodestyle pylint codecov coverage

script:
# Build the conda package
- git fetch origin --tags
- conda build -q tools/conda.recipe

# Build source package, should work too and needed for deployment to Github and
# PyPI.
- python setup.py sdist

# Install Conda package
- conda install --use-local $PROJECT_NAME

# Many options are added to get a thorough coverage analysis.
- "(cd; nosetests ${PROJECT_NAME} -v --processes= --detailed-errors --with-coverage --cover-package=${PROJECT_NAME} --cover-tests --cover-erase --cover-inclusive --cover-branches --cover-xml)"

# Code quality checks
- pycodestyle derivcheck/__init__.py derivcheck/basic_example.py setup.py
- pycodestyle --ignore=E731 derivcheck/test_derivcheck.py
- pydocstyle setup.py derivcheck
- pylint derivcheck/__init__.py derivcheck/test_derivcheck.py derivcheck/basic_example.py

after_success:
# Upload the coverage analysis
- codecov -f ~/coverage.xml

# For deployment, the env var TRAVIS_TAG contains the name of the current tag, if any.
before_deploy:
- conda convert --platform all $HOME/miniconda/conda-bld/linux-64/${PROJECT_NAME}-*.tar.bz2 --output-dir conda-bld/
- cp $HOME/miniconda/conda-bld/linux-64/${PROJECT_NAME}-*.tar.bz2 conda-bld/

deploy:
- provider: releases
  skip_cleanup: true
  api_key: ${GITHUB_TOKEN}
  file: dist/${PROJECT_NAME}-${TRAVIS_TAG}.tar.gz
  on:
    repo: ${GITHUB_REPO_NAME}
    tags: true
    condition: "$TRAVIS_TAG != *[ab]* && $MYCONDAPY == 2.7 && $TRAVIS_OS_NAME == linux"
  prerelease: false
- provider: releases
  skip_cleanup: true
  api_key: ${GITHUB_TOKEN}
  file: dist/${PROJECT_NAME}-${TRAVIS_TAG}.tar.gz
  on:
    repo: ${GITHUB_REPO_NAME}
    tags: true
    condition: "$TRAVIS_TAG == *[ab]* && $MYCONDAPY == 2.7 && $TRAVIS_OS_NAME == linux"
  prerelease: true
- provider: script
  skip_cleanup: true
  script: anaconda -t $ANACONDA_TOKEN upload --force -l alpha conda-bld/**/${PROJECT_NAME}-*.tar.bz2
  on:
    repo: ${GITHUB_REPO_NAME}
    tags: true
    condition: "$TRAVIS_TAG == *a* && $TRAVIS_OS_NAME == linux"
- provider: script
  skip_cleanup: true
  script: anaconda -t $ANACONDA_TOKEN upload --force -l beta conda-bld/**/${PROJECT_NAME}-*.tar.bz2
  on:
    repo: ${GITHUB_REPO_NAME}
    tags: true
    condition: "$TRAVIS_TAG == *b* && $TRAVIS_OS_NAME == linux"
- provider: script
  skip_cleanup: true
  script: anaconda -t $ANACONDA_TOKEN upload --force conda-bld/**/${PROJECT_NAME}-*.tar.bz2
  on:
    repo: ${GITHUB_REPO_NAME}
    tags: true
    condition: "$TRAVIS_TAG != *[ab]* && $TRAVIS_OS_NAME == linux"
- provider: pypi
  skip_cleanup: true
  user: theochem
  password: ${PYPI_PASSWD}
  on:
    repo: ${GITHUB_REPO_NAME}
    tags: true
    condition: "$TRAVIS_TAG != *[ab]* && $MYCONDAPY == 2.7 && $TRAVIS_OS_NAME == linux"
