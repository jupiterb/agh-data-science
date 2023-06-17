import numpy as np


class FlopsCounter:
    def __init__(self) -> None:
        self._flops = 0

    @property
    def flops(self):
        return self._flops

    def reset_flops(self):
        self._flops = 0


def get_submatrices(M: np.ndarray):
    assert (
        not M.shape[0] % 2 and not M.shape[1] % 2
    ), f"Unsupported matrix shape {M.shape}"
    half_m, half_n = M.shape[0] // 2, M.shape[1] // 2
    return (
        M[:half_m, :half_n],
        M[:half_m:, half_n:],
        M[half_m:, :half_n],
        M[half_m:, half_n:],
    )
