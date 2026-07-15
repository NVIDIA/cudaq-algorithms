def test_import():
    import cudaq_algorithms

    assert "CUDA-Q Algorithms" in cudaq_algorithms.__version__
