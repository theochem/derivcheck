#!/usr/bin/env bash

# Token howto:
# anaconda login
# anaconda auth -c -n travis --max-age 307584000 --url https://anaconda.org/USERNAME/PACKAGENAME --scopes "api:write api:read"
# travis encrypt SOMEVAR="secretvalue"

LABEL=${1}
set -e

echo "Converting conda package..."
conda convert --platform all $HOME/miniconda2/conda-bld/linux-64/derivcheck-*.tar.bz2 --output-dir conda-bld/

echo "Deploying to Anaconda.org..."
anaconda -t $ANACONDA_TOKEN upload -l ${LABEL} conda-bld/**/derivcheck-*.tar.bz2

echo "Successfully deployed to Anaconda.org."
exit 0
