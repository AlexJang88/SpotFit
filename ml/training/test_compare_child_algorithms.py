from compare_child_algorithms import improved_mae


def test_improved_mae_requires_strict_reduction():
    assert improved_mae(9.9, 10.0)
    assert not improved_mae(10.0, 10.0)
    assert not improved_mae(10.1, 10.0)


def test_improved_mae_honors_minimum_improvement():
    assert improved_mae(9.8, 10.0, minimum=0.1)
    assert not improved_mae(9.95, 10.0, minimum=0.1)
