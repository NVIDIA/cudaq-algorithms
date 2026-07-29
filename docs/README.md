# Building the CUDA-Q Algorithms documentation

The docs are a [Sphinx](https://www.sphinx-doc.org/) site (Read-the-Docs
theme) with narrative guides plus an autodoc-generated API reference. The
sources live under `docs/sphinx/`.

## Requirements

Sphinx autodoc **imports** the `cudaq_algorithms` package to read its
docstrings, and the package imports `cudaq` — so the interpreter that runs
Sphinx must have `cudaq` importable. You do **not** need to build CUDA-Q from
source; a released wheel is enough (autodoc only imports the modules, it never
runs a kernel, so no GPU or CUDA toolkit is required):

```bash
python3 -m pip install cudaq                     # released wheel (numpy/scipy come with it)
python3 -m pip install -r docs/requirements.txt  # Sphinx + theme + myst/copybutton/tabs
```

Install `cudaq`, Sphinx, and the doc requirements into the **same** interpreter
(a virtualenv is the easy way to guarantee that).

## Build

From the repository root:

```bash
PYTHONPATH="$PWD/python" make -C docs html
```

- `PYTHONPATH="$PWD/python"` makes the local (checked-out, not pip-installed)
  `cudaq_algorithms` package importable for autodoc. `cudaq` is found via the
  pip install above.
- The build runs with `-W` (warnings treated as errors), so a clean exit means
  the docs are genuinely clean; any warning fails the build.
- If `python3` is not the interpreter you installed into, override the builder:

  ```bash
  PYTHONPATH="$PWD/python" make -C docs html SPHINXBUILD="python3.12 -m sphinx"
  ```

The generated site is written to `docs/sphinx/_build/html/`.

## View

On a machine with a browser:

```bash
open docs/sphinx/_build/html/index.html      # macOS ("xdg-open" on Linux)
```

On a headless or remote host, serve it and browse over the network:

```bash
python3 -m http.server 8000 --directory docs/sphinx/_build/html
# then open http://<host>:8000
```

## Clean

```bash
make -C docs clean
```

## Notes

- Narrative pages are reStructuredText under `docs/sphinx/guide/`,
  `docs/sphinx/getting_started/`, `docs/sphinx/examples_rst/`, and
  `docs/sphinx/conventions.rst`; the API reference is
  `docs/sphinx/api/python_api.rst` (autodoc).
- Example scripts live under `docs/sphinx/examples/python/` and are pulled into
  the example pages with `.. literalinclude::` sliced at a
  `# [Begin Documentation]` marker — edit the script, not a copy in the prose.
- CI builds and (on release branches) publishes the site via
  `.github/workflows/docs.yaml`.
