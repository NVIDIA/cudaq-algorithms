"""Guard pure-host tests without constraining unrelated native tests."""
import sys
import pytest


@pytest.fixture(autouse=True)
def no_cudaq_import():
    # Native tests can legitimately import CUDA-Q during combined collection.
    # Host tests must not introduce it; the subprocess test below also proves
    # purity in an interpreter where no native test has run.
    cudaq_was_loaded = 'cudaq' in sys.modules
    yield
    assert cudaq_was_loaded or 'cudaq' not in sys.modules
