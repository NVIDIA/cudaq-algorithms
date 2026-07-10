def test_import():
    import cudaq_algorithms

    assert "CUDA-Q Algorithms" in cudaq_algorithms.__version__


def test_native_extension_state_is_consistent():
    # If the compiled extension is present it must have loaded (a broken
    # extension re-raises at package import); the silent fallback is only
    # legitimate when the module is genuinely absent.
    import importlib.util

    import cudaq_algorithms

    spec = importlib.util.find_spec("cudaq_algorithms._pycudaq_algorithms")
    if spec is None:
        assert cudaq_algorithms._NATIVE_IMPORT_ERROR is not None
    else:
        assert cudaq_algorithms._NATIVE_IMPORT_ERROR is None
