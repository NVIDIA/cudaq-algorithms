# Minimal Sphinx build for the cudaq-algorithms docs.
#
# autodoc imports the package, which imports `cudaq`, so the build env must
# have cudaq (+ numpy/scipy) importable, e.g.:
#   export LD_LIBRARY_PATH=<cudaq>/lib
#   export PYTHONPATH=<cudaq>:$(pwd)/../python
#   make -C docs html

SPHINXBUILD ?= python3 -m sphinx
# Warnings-as-errors (nitpicky --keep-going surfaces them all), matching cudaqx.
SPHINXOPTS  ?= -W --keep-going
SOURCEDIR    = sphinx
BUILDDIR     = sphinx/_build

.PHONY: html clean

html:
	$(SPHINXBUILD) $(SPHINXOPTS) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"
	@echo "Built docs in $(BUILDDIR)/html/index.html"

clean:
	rm -rf "$(BUILDDIR)"
