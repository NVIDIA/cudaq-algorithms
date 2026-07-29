# Minimal Sphinx build for the cudaq-algorithms docs.
#
# autodoc imports the package, which imports `cudaq`, so the build env must
# have cudaq (+ numpy/scipy) importable, e.g.:
#   export LD_LIBRARY_PATH=<cudaq>/lib
#   export PYTHONPATH=<cudaq>:$(pwd)/../python
#   make -C docs html

SPHINXBUILD ?= python3 -m sphinx
# Target is `-W --keep-going` (warnings-as-errors, matching cudaqx). Kept at
# --keep-going until the autodoc docstring nits (bare `|` ket notation) and
# cross-references are cleaned up, then flip -W on.
SPHINXOPTS  ?= --keep-going
SOURCEDIR    = sphinx
BUILDDIR     = sphinx/_build

.PHONY: html clean

html:
	$(SPHINXBUILD) $(SPHINXOPTS) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"
	@echo "Built docs in $(BUILDDIR)/html/index.html"

clean:
	rm -rf "$(BUILDDIR)"
