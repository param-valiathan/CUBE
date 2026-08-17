"""Phase T1: confirms pytest collects and runs cleanly via the CUBE env."""
import cube_core


def test_cube_core_imports():
    assert hasattr(cube_core, "BSoidEngine")


def test_pytest_infra_sane():
    assert 1 + 1 == 2
